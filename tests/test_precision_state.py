import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from transformers import DynamicCache

from qaq.model import set_profile
from qaq.quantization import reconstruct
from precision_state.harness import (CachedRunner, Segment, cache_tensors, free_schedules,
                                     kv_metadata, tiny_model, token_hash, validate_schedule)
from precision_state.artifacts import new_run, save_array, seal, verify, write_json
from precision_state.protocol import select_prompts, authorize_main
from precision_state.analysis import (bootstrap_mean, decide, first_divergence, free_gate,
                                      history_gate, logit_metrics, paired_free)


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
TEST_TMP = Path(__file__).resolve().parents[1]/"raw/test-tmp"
TEST_TMP.mkdir(parents=True, exist_ok=True)


class AConstantControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = tiny_model()
        cls.runner = CachedRunner(cls.model)
        cls.ids = [1, 7, 9, 2, 15, 6, 32, 4, 11, 17, 18, 45, 3, 42, 51, 27, 16]

    def test_00_constant_w4_and_w8_chunking_before_history(self):
        # FP16 unit-scale accumulation bound: 2 ULP at magnitude~1. Float32 CPU
        # logit conversion is lossless. This tiny-only tolerance is NOT a main gate.
        for bits in (8, 4):
            arrays, traces = self.runner.constant_control(self.ids, bits, 8)
            torch.testing.assert_close(arrays["oneshot"], arrays["chunked"], rtol=0, atol=2e-3)
            self.assertTrue(torch.equal(arrays["chunked"], arrays["repeat"]))
            for trace in traces:
                self.assertEqual([t["cache_length_before"] for t in trace], [0, 8, 16])
                self.assertEqual([t["cache_length_after"] for t in trace], [8, 16, 17])
                self.assertEqual(len({t["cache_id_in"] for t in trace}), 1)
                self.assertTrue(all(t["cache_id_in"] == t["cache_id_out"] and t["prefix_preserved"] for t in trace))

    def test_actual_switches_reconstruct_the_requested_weights(self):
        for order in ([8, 4], [4, 8], [8, 8], [4, 4]):
            logits, trace = self.runner.teacher(self.ids, order, 8)
            self.assertEqual(logits.shape, (64,))
            self.assertEqual([t["requested_bits"] for t in trace], order + [8])
            for t, bits in zip(trace, order + [8]):
                self.assertEqual(set(t["applied_bits"].values()), {bits})
                self.assertEqual(len(t["applied_bits"]), 14)
            self.assertEqual([i for t in trace for i in t["token_ids"]], self.ids)
            for m in self.runner.modules.values():
                self.assertTrue(torch.equal(m.active_weight, reconstruct(m.q, m.scale, 8, torch.float16)))
        # W8 and W4 compute really differ, not just a recorded metadata switch.
        a, _ = self.runner.sequence(self.ids, [Segment(17, 8)])
        b, _ = self.runner.sequence(self.ids, [Segment(17, 4)])
        self.assertGreater((a-b).abs().max().item(), 0)

    def test_cached_history_is_not_recomputed(self):
        cache = DynamicCache(config=self.model.config)
        self.runner.forward(self.ids[:8], 8, cache)
        previous = [(k.clone(), v.clone()) for k, v in cache_tensors(cache)]
        _, trace = self.runner.forward(self.ids[8:16], 4, cache)
        self.assertTrue(trace["prefix_preserved"])
        for old, new in zip(previous, cache_tensors(cache)):
            for a, b in zip(old, new):
                self.assertTrue(torch.equal(a, b[..., :8, :]))
        _, recomputed_trace = self.runner.sequence(self.ids[:16], [Segment(16, 4)])
        self.assertNotEqual(trace["kv_after_sha256"], recomputed_trace[0]["kv_after_sha256"])

    def test_schedule_mismatch_is_detected_at_projection_forward(self):
        with patch("precision_state.harness.set_profile", lambda *_: set_profile(self.model, [4]*4)):
            with self.assertRaisesRegex(AssertionError, "applied schedule"):
                self.runner.sequence(self.ids, [Segment(17, 8)])

    def test_invalid_schedule_does_not_mutate_model(self):
        before = [m.bits for m in self.runner.modules.values()]
        with self.assertRaises(ValueError):
            self.runner.sequence(self.ids, [Segment(8, 4), Segment(9, 5)])
        self.assertEqual(before, [m.bits for m in self.runner.modules.values()])

    def test_deterministic_free_run_and_kv_lengths(self):
        for name, schedule in free_schedules(2).items():
            a = self.runner.generate(self.ids[:4], schedule, [63])
            b = self.runner.generate(self.ids[:4], schedule, [63])
            self.assertEqual(a["token_ids"], b["token_ids"])
            self.assertEqual(first_divergence(a["token_ids"], b["token_ids"]), None)
            n = a["output_length"]
            self.assertEqual(sum(a["decode_exposure"].values()), n)
            self.assertEqual(sum(a["total_exposure"].values()), 3+n)
            self.assertEqual(a["trace"][-1]["cache_length_after"], 3+n)
            self.assertEqual(a["length_cap_hit"], n == 4)
            expected_bytes_per_token = 2 * 2 * 2 * 32 * 2  # layer*K/V*kv_heads*dim*FP16
            expected = [3*expected_bytes_per_token] + [(4+i)*expected_bytes_per_token for i in range(n)]
            self.assertEqual([t["kv"]["storage_bytes"] for t in a["trace"]], expected)
            self.assertEqual(a["cumulative_kv_byte_forwards"], sum(expected))
            self.assertEqual(a["peak_kv_storage_bytes"], max(expected))
            self.assertEqual(paired_free(a, b)["length_delta"], 0)

    def test_eos_before_switch_retained_and_labelled(self):
        schedule = free_schedules(2)["8to4"]
        baseline = self.runner.generate(self.ids[:4], schedule, [63])
        eos = baseline["token_ids"][0]
        a = self.runner.generate(self.ids[:4], schedule, [eos])
        self.assertTrue(a["eos"])
        self.assertEqual(a["token_ids"], [eos])
        self.assertEqual(a["output_length"], 1)
        self.assertFalse(a["equal_decode_exposure"])
        self.assertFalse(a["length_cap_hit"])
        self.assertEqual(a["decode_exposure"], {8: 1})
        self.assertEqual(a["trace"][-1]["cache_length_after"], 4)  # EOS not cached


class ScheduleAndBytesTests(unittest.TestCase):
    def test_validation_and_equal_counts(self):
        for order in ([8, 4], [4, 8]):
            s = [Segment(8, order[0]), Segment(8, order[1]), Segment(1, 8)]
            self.assertEqual(validate_schedule(s, 17, balanced=True, probe=True), {4: 8, 8: 8})
        bad = [[], [Segment(0, 8)], [Segment(-1, 4)], [Segment(1, 5)], [Segment(True, 4)],
               [Segment(1, True)], [Segment(1, 8.0)], [Segment(2, 8)]]
        for s in bad:
            with self.assertRaises(ValueError):
                validate_schedule(s, 1)
        with self.assertRaises(ValueError):
            validate_schedule([Segment(7, 8), Segment(9, 4), Segment(1, 8)], 17, balanced=True, probe=True)
        with self.assertRaises(ValueError):
            validate_schedule([Segment(8, 8), Segment(8, 4), Segment(1, 4)], 17, probe=True)
        with self.assertRaises(ValueError):
            validate_schedule([Segment(8, 6), Segment(8, 6), Segment(1, 8)], 17, balanced=True, probe=True)

    def test_exact_backing_and_logical_bytes_including_shared_views(self):
        cache = DynamicCache()
        backing = torch.zeros((1, 2, 9, 4), dtype=torch.float32)
        cache.update(backing[..., :3, :], backing[..., 3:6, :], 0)
        # DynamicCache.update concatenates even on the first call (copies views).
        # Explicitly install shared views to exercise logical-vs-backing accounting.
        cache.layers[0].keys = backing[..., :3, :]
        cache.layers[0].values = backing[..., 3:6, :]
        info = kv_metadata(cache)
        self.assertEqual(info["tensor_bytes"], 2*1*2*3*4*4)
        self.assertEqual(info["storage_bytes"], backing.untyped_storage().nbytes())
        self.assertEqual([t["storage_group"] for t in info["tensors"]], [0, 0])
        self.assertEqual(kv_metadata(DynamicCache())["storage_bytes"], 0)


class ArtifactAndSelectionTests(unittest.TestCase):
    def test_append_only_results_and_lossless_arrays(self):
        with tempfile.TemporaryDirectory(dir=TEST_TMP) as d:
            out = new_run(Path(d)/"run")
            write_json(out/"record.json", {"x": 1})
            array = np.array([1, -2, 0.25], dtype=np.float32)
            save_array(out/"logits.npy", array)
            np.testing.assert_array_equal(np.load(out/"logits.npy", allow_pickle=False), array)
            with self.assertRaises(FileExistsError):
                new_run(out)
            with self.assertRaises(FileExistsError):
                write_json(out/"record.json", {})
            with self.assertRaises(FileExistsError):
                save_array(out/"logits.npy", array)
            seal(out)
            verify(out)
            with self.assertRaises(FileExistsError):
                seal(out)
            (out/"logits.npy").write_bytes(b"tamper")
            with self.assertRaises(ValueError):
                verify(out)

    def test_selection_identical_tokens_and_disjoint_ids(self):
        examples = [{"task": "wikitext2", "index": n, "tokens": [n, 2, 3, 4, 5]} for n in range(40)]
        a = select_prompts(examples, chunk=2)
        b = select_prompts(list(reversed(examples)), chunk=2)
        self.assertEqual(a, b)
        self.assertEqual(len(a["main"]), 32)
        self.assertEqual(len(a["pilot"]), 4)
        self.assertFalse({r["id"] for r in a["main"]} & {r["id"] for r in a["pilot"]})
        for r in a["main"]:
            self.assertEqual(token_hash(r["token_ids"]), r["token_sha256"])
        with self.assertRaises(ValueError):
            select_prompts(examples, main_n=15, chunk=2)
        with self.assertRaises(ValueError):
            select_prompts(examples, pilot_n=5, chunk=2)
        with self.assertRaises(ValueError):
            select_prompts(examples+[examples[0]], chunk=2)
        with self.assertRaises(ValueError):
            authorize_main({"status": "awaiting_pilot"}, ".")


class AnalysisAndBoundaryTests(unittest.TestCase):
    def test_metrics_known_distributions_ties_and_shift(self):
        m = logit_metrics([0., 0., -1000.], [0., -1000., 0.], k=2)
        self.assertEqual(m["total_variation"], 0.5)
        self.assertEqual(m["topk_overlap"], 0.5)
        self.assertFalse(m["top1_changed"])
        self.assertAlmostEqual(m["js_nats"], np.log(2)/2)
        m = logit_metrics([1., 2.], [11., 12.], k=1)
        self.assertEqual(m["max_abs_logit_diff"], 10)
        self.assertAlmostEqual(m["total_variation"], 0)
        with self.assertRaises(ValueError):
            logit_metrics([np.nan, 0], [0, 0], k=1)

    def test_divergence_eos_length_and_none(self):
        self.assertEqual(first_divergence([1, 2, 3], [1, 8, 3]), 1)
        self.assertEqual(first_divergence([1], [1, 2]), 1)
        self.assertIsNone(first_divergence([1, 2], [1, 2]))
        self.assertEqual(first_divergence([], [1]), 0)

    def test_bootstrap_is_deterministic_and_paired(self):
        self.assertEqual(bootstrap_mean([1, 2, -3]), bootstrap_mean([1, 2, -3]))
        r = bootstrap_mean([2]*32)
        self.assertEqual([r[k] for k in ("mean", "lo", "hi")], [2, 2, 2])
        with self.assertRaises(ValueError):
            bootstrap_mean([])

    def test_decision_boundaries_and_minimum(self):
        base = dict(n=32, frozen=True, valid=True, fresh_processes=2, repeat_agreement=True,
                    history_pass=False, free_pass=False)
        for h, f, expected in [(False, False, "stop_precision_state_direction"),
                               (True, False, "continue_precision_provenance"),
                               (False, True, "revise_endogenous_workload"),
                               (True, True, "continue_joint_state_work")]:
            for n in (16, 32):
                self.assertEqual(decide(**{**base, "n": n, "history_pass": h, "free_pass": f}), expected)
        for key, value in [("n", 15), ("n", 0), ("frozen", False), ("valid", False),
                           ("fresh_processes", 1), ("fresh_processes", 3), ("repeat_agreement", False)]:
            self.assertEqual(decide(**{**base, key: value, "history_pass": True, "free_pass": True}), "pause_blocked")
        with self.assertRaises(ValueError):
            decide(**{**base, "valid": None})

    def test_gates_strict_threshold_no_posthoc_equal_pass(self):
        rows = [{"max_abs_logit_diff": 0.2, "total_variation": 0.01}]*32
        self.assertFalse(history_gate(rows, 0.1, 0.01)["passes"])
        self.assertFalse(history_gate(rows, 0.2, 0.005)["passes"])
        self.assertTrue(history_gate(rows, 0.1, 0.005)["passes"])
        rows = [{"abs_length_delta": 4, "eos_discordant": 0.125, "abs_relative_cumulative_kv_delta": 0.1}]*32
        self.assertFalse(free_gate(rows, 4, 0.125, 0.1)["passes"])
        self.assertTrue(free_gate(rows, 3, 0.125, 0.1)["passes"])


if __name__ == "__main__":
    unittest.main()
