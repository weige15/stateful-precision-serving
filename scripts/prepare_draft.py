"""Choose IDs solely from already-available token records; does NOT freeze v1."""
import json
from pathlib import Path
from precision_state.artifacts import sha256, write_json
from precision_state.protocol import select_prompts

ROOT = Path(__file__).resolve().parents[1]


def main():
    intake_path = ROOT/"raw/intake-002/intake.json"
    source = ROOT.parent/"qaq_baseline/results/core-v1/frozen/examples.jsonl"
    intake = json.loads(intake_path.read_text())
    if sha256(source) != intake["inputs"][str(source)]["sha256"]:
        raise ValueError("input records changed")
    examples = [json.loads(s) for s in source.read_text().splitlines()]
    config = {
        "protocol_id": "precision-state-v1", "status": "awaiting_real_pilot_NOT_FROZEN",
        "authorized_fresh_main_processes": 0, "pilot_controls_passed": False, "runtime_authorized": False,
        "intake_path": str(intake_path.relative_to(ROOT)), "intake_sha256": sha256(intake_path),
        "upstream_commit": intake["upstream_commit"],
        "checkpoint": str(ROOT.parent/"qaq_baseline/results/core-v1/integration/quantized_model.pt"),
        "model_id": "Qwen/Qwen3-4B", "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "tokenizer_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "input_examples": str(source), "input_examples_sha256": sha256(source),
        "formatting": "provisional: existing wikitext2 token IDs verbatim; no chat template, no retokenization, no added special tokens",
        "selection": "sort all existing 64 WT2 candidates by SHA256(canonical JSON [1729, task:index]); first4 pilot, next32 main",
        "prompts": select_prompts(examples), "chunk_tokens": 64, "history_tokens": 128, "probe_tokens": 1,
        "probe_definition": "input token_ids[128] under W8; logits predict the unconsumed token at position129",
        "teacher_schedules": {"84": [8, 4, 8], "48": [4, 8, 8], "88": [8, 8, 8], "44": [4, 4, 8]},
        "teacher_segment_lengths": [64, 64, 1], "fresh_cache_repeats_per_schedule": 2,
        "constant_controls": "all129 tokens under W4 or W8, use_cache=False one-shot vs cached64+64+1, two fresh cached repeats",
        "decode_segment_tokens": 16, "max_new_tokens": 32,
        "free_schedules": {"fixed4": [[32, 4]], "fixed6": [[32, 6]], "fixed8": [[32, 8]], "4to8": [[16, 4], [16, 8]], "8to4": [[16, 8], [16, 4]]},
        "free_prefill": "first128 prompt tokens under schedule's first precision; remaining prompt token starts decode; prefill excluded from 16+16 segment counts but included in total exposure and KV accounting",
        "decoding": {"do_sample": False, "algorithm": "argmax; lowest token ID breaks exact ties", "batch_size": 1,
                     "beam_size": 1, "temperature": None, "top_k": None, "top_p": None, "repetition_penalty": None},
        "eos_ids": [151645, 151643], "seed": 1729, "torch_deterministic_algorithms": True,
        "attention": "sdpa", "dtype": "float16", "allow_tf32": False,
        "pilot_control_ceiling": 0.125,
        "pilot_control_ceiling_note": "provisional fail-closed engineering ceiling (128 FP16 ULP at1), not a noise gate; passing alone cannot establish main validity",
        "constant_control_tolerance": None, "numerical_noise_gate": None,
        "noise_gate_rule_proposed": "freeze max(1e-6,10*max pilot repeat/chunk max_abs over both bits, all4 prompts and all positions); exact repeats expected. Main controls must remain within frozen pilot envelope plus1 FP16 ULP at observed max|logit|; any unexplained discrepancy pauses",
        "practical_history_tv_gate_proposed": 0.01,
        "practical_free_gates_proposed": {"mean_abs_length_delta": 4, "eos_discordance": 0.125, "mean_abs_relative_cumulative_kv_delta": 0.1},
        "uncertainty_proposed": {"unit": "prompt (not repeat)", "method": "paired percentile bootstrap mean, linear quantile",
                                 "seed": 314159, "draws": 10000, "history_confidence": 0.95,
                                 "free_confidence": 0.9958333333333333,
                                 "free_multiplicity": "Bonferroni across4 comparator-vs-fixed8 contrasts x3 practical endpoints"},
        "main_target": 32, "main_minimum": 16, "pilot_maximum": 4,
        "exclusions": "No outcome exclusions. Bad hashes, nonfinite logits, missing schedules, broken caches, unexplained controls or insufficient budget invalidate/pause. Retain EOS-before-switch and cap hits, no equal total-bit claims.",
        "repeat_agreement": "two fresh main processes must have exact free token paths/EOS/lengths and schedule exposure; full teacher logits within frozen numerical gate and same history/free branch; report every discrepancy",
        "gpu_limits": {"serial": True, "process_free": True, "timeout_seconds": 1800, "goal_seconds": 14400},
        "frozen_files": {}, "decision_rules_status": "proposed in PRECISION_STATE_PROTOCOL.md; NOT authorized"
    }
    (ROOT/"configs").mkdir(exist_ok=True)
    write_json(ROOT/"configs/precision_state_v1.json", config)
    print("Draft written: 4 pilot and32 main IDs; NOT frozen, no main authorization")


if __name__ == "__main__":
    main()
