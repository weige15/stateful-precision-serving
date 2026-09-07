import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from scripts import gpu_job, audit_tiny
from precision_state.analysis import bootstrap_mean, logit_metrics
from precision_state.protocol import authorize_main
from precision_state.harness import CachedRunner, tiny_model

ROOT = Path(__file__).resolve().parents[1]
(ROOT/"raw/test-tmp").mkdir(parents=True, exist_ok=True)


def temporary():
    return tempfile.TemporaryDirectory(dir=ROOT/"raw/test-tmp")


def xml(processes="", free="24000 MiB", used="10 MiB", util="0 %"):
    return f"<nvidia_smi_log><gpu><uuid>GPU-test</uuid><product_name>test</product_name><fb_memory_usage><free>{free}</free><used>{used}</used></fb_memory_usage><utilization><gpu_util>{util}</gpu_util></utilization><processes>{processes}</processes></gpu></nvidia_smi_log>"


class SafetyTests(unittest.TestCase):
    def test_gpu_selection_excludes_any_process_and_unknowns(self):
        self.assertEqual(gpu_job.select_idle(xml())[0], "GPU-test")
        for snapshot in [xml(processes="<process_info><pid>99</pid><type>G</type></process_info>"),
                         xml(processes="N/A"), xml(free="N/A"), xml(util="1 %"), xml(used="2000 MiB"),
                         xml(free="20000 MiB"), xml().replace("<processes></processes>", "")]:
            self.assertIsNone(gpu_job.select_idle(snapshot)[0])

    def test_gpu_refusal_creates_no_reservation_or_launch(self):
        with temporary() as d:
            root = Path(d)
            out = root/"raw/gpu-jobs/refused"
            response = SimpleNamespace(returncode=0, stdout=xml(util="100 %"), stderr="")
            with patch.object(gpu_job, "ROOT", root), patch.object(sys, "argv", ["gpu_job.py", "--stage", "pilot", "--out", str(out)]), patch.object(gpu_job.subprocess, "run", return_value=response) as run:
                self.assertEqual(gpu_job.main(), 3)
                self.assertEqual(run.call_count, 1)
            self.assertFalse((out/"reservation.json").exists())
            self.assertFalse((out/"launch.json").exists())
            self.assertEqual(gpu_job.charged_seconds(out.parent), 0)

    def test_gpu_lock_seriality_and_fail_closed_ledger(self):
        import fcntl
        with temporary() as d:
            path = Path(d)/"lock"
            with path.open("a") as a, path.open("a") as b:
                fcntl.flock(a, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(b, fcntl.LOCK_EX | fcntl.LOCK_NB)
            job = Path(d)/"failed"
            job.mkdir()
            (job/"reservation.json").write_text(json.dumps({"reserved_seconds": 1860}))
            self.assertEqual(gpu_job.charged_seconds(d), 1860)
            (job/"completion.json").write_text(json.dumps({"charged_seconds": 12.5, "returncode": 1}))
            self.assertEqual(gpu_job.charged_seconds(d), 12.5)
            (job/"completion.json").write_text(json.dumps({"charged_seconds": -1}))
            with self.assertRaises(ValueError):
                gpu_job.charged_seconds(d)

    def test_guard_timeout_and_completed_failure_charge_without_gpu(self):
        with temporary() as d:
            root = Path(d)
            (root/"configs").mkdir()
            (root/"configs/precision_state_v1.json").write_text("{}")
            out = root/"raw/gpu-jobs/mock-timeout"
            responses = [SimpleNamespace(returncode=0, stdout=xml(), stderr="")]*2
            with patch.object(gpu_job, "ROOT", root), patch.object(sys, "argv", ["gpu_job.py", "--stage", "pilot", "--out", str(out)]), patch.object(gpu_job.subprocess, "run", side_effect=responses), patch.object(gpu_job, "execute_owned", return_value=(124, True)) as execute, patch.object(gpu_job.time, "sleep"), patch.object(gpu_job.time, "monotonic", side_effect=[0, 1800]):
                self.assertEqual(gpu_job.main(), 124)
                self.assertEqual(execute.call_args.args[0][:4], ["timeout", "--signal=TERM", "--kill-after=5s", "1790s"])
            completion = json.loads((out/"completion.json").read_text())
            self.assertTrue(completion["timeout"])
            self.assertEqual(gpu_job.charged_seconds(out.parent), 1800)

    def test_pending_job_and_budget_prevent_launch(self):
        for finished in (False, True):
            with temporary() as d:
                root = Path(d)
                prior = root/"raw/gpu-jobs/previous"
                prior.mkdir(parents=True)
                (prior/"reservation.json").write_text(json.dumps({"reserved_seconds": 1860}))
                if finished:
                    (prior/"completion.json").write_text(json.dumps({"charged_seconds": 14000}))
                out = prior.parent/"new"
                response = SimpleNamespace(returncode=0, stdout=xml(), stderr="")
                with patch.object(gpu_job, "ROOT", root), patch.object(sys, "argv", ["gpu_job.py", "--stage", "pilot", "--out", str(out)]), patch.object(gpu_job.subprocess, "run", return_value=response) as run, patch.object(gpu_job.time, "sleep"):
                    self.assertEqual(gpu_job.main(), 3)
                    self.assertEqual(run.call_count, 2)  # preflight only
                self.assertFalse((out/"launch.json").exists())

    def test_owned_timeout_stops_child_group_cpu_only(self):
        with temporary() as d:
            with (Path(d)/"log").open("x") as log:
                started = time.monotonic()
                result = gpu_job.execute_owned([sys.executable, "-c", "import time; time.sleep(60)"], cwd=ROOT, env=os.environ.copy(), log=log, timeout_seconds=5.05)
                self.assertEqual(result, (124, True))
                self.assertLess(time.monotonic()-started, 5)

    def test_wrapper_only_sigint_does_not_orphan_grandchild(self):
        with temporary() as d:
            root = Path(d)
            pid_file = root/"grandchild.pid"
            workload = "import subprocess,sys,time; from pathlib import Path; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(60)"
            wrapper = "import os,sys; from scripts.gpu_job import execute_owned; result=execute_owned(sys.argv[1:],cwd=os.getcwd(),env=os.environ.copy(),log=sys.stdout); sys.exit(result[0])"
            with (root/"wrapper.log").open("x") as log:
                p = subprocess.Popen([sys.executable, "-c", wrapper, "timeout", "--signal=TERM", "--kill-after=5s", "30s", sys.executable, "-c", workload, str(pid_file)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                try:
                    deadline = time.monotonic()+10
                    while not pid_file.exists() and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(pid_file.exists(), "dummy grandchild did not start")
                    grandchild = int(pid_file.read_text())
                    os.kill(p.pid, signal.SIGINT)  # only wrapper, not owned workload group
                    self.assertEqual(p.wait(timeout=8), 130)
                    status = Path(f"/proc/{grandchild}/status")
                    if status.exists():
                        state = next(s for s in status.read_text().splitlines() if s.startswith("State:"))
                        self.assertIn("Z", state, "grandchild remains live after wrapper interruption")
                finally:
                    if p.poll() is None:
                        p.kill()
                        p.wait()

    def test_draft_cannot_authorize_main(self):
        config = json.loads((ROOT/"configs/precision_state_v1.json").read_text())
        self.assertEqual(len(config["prompts"]["main"]), 32)
        with self.assertRaisesRegex(ValueError, "not frozen"):
            authorize_main(config, ROOT)
        self.assertFalse((ROOT/"evidence/freeze.json").exists())


class IndependentCrossChecks(unittest.TestCase):
    def test_metric_implementation_independent(self):
        rng = np.random.default_rng(42)
        for _ in range(10):
            a, b = rng.normal(size=(2, 64)).astype(np.float32)
            audit_tiny.near(audit_tiny.metrics(a, b), logit_metrics(a, b))

    def test_bootstrap_independently_sorted_quantiles(self):
        values, seed, draws, confidence = [1, -3, 7, 0, 2], 101, 1000, 0.95
        rng = np.random.default_rng(seed)
        means = sorted(sum(values[int(i)] for i in rng.integers(len(values), size=len(values)))/len(values) for _ in range(draws))
        def q(probability):
            location = probability*(draws-1)
            low = int(location)
            return means[low] + (means[min(low+1, draws-1)]-means[low])*(location-low)
        actual = bootstrap_mean(values, seed=seed, draws=draws, confidence=confidence)
        self.assertAlmostEqual(actual["lo"], q(0.025))
        self.assertAlmostEqual(actual["hi"], q(0.975))

    def test_tiny_real_length_constant_controls(self):
        runner = CachedRunner(tiny_model())
        ids = [(i*7)%62+1 for i in range(129)]
        for bits in (8, 4):
            arrays, _ = runner.constant_control(ids, bits, 64)
            self.assertLessEqual(float((arrays["oneshot"]-arrays["chunked"]).abs().max()), 0.002)
            self.assertTrue(np.array_equal(arrays["chunked"].numpy(), arrays["repeat"].numpy()))

    def test_audit_rejects_optimized_python_and_missing_analysis_runs(self):
        p = subprocess.run([sys.executable, "-O", "scripts/audit_tiny.py", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=20)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("unoptimized Python", p.stderr)
        with self.assertRaises(ValueError):
            audit_tiny.validate_analysis({"runs": [], "decision": "pause_blocked"}, [ROOT, ROOT], [[], []])
        with self.assertRaises(ValueError):
            audit_tiny.validate_analysis({"runs": [{"run": "/wrong", "rows": []}]*2}, [ROOT, ROOT], [[], []])

    def test_audit_rejects_teacher_tokens_and_schedule_tampering(self):
        root = ROOT/"raw/tiny-001/data"
        row = json.loads((root/"prompt-0.json").read_text())
        cfg = json.loads((root/"model.json").read_text())["config"]
        original = row["teacher"][0]["trace"]
        for edit in ("token", "bits", "bytes", "cache", "prefix"):
            trace = copy.deepcopy(original)
            if edit == "token":
                trace[1]["token_ids"][0] += 1
            elif edit == "bits":
                key = next(iter(trace[1]["applied_bits"]))
                trace[1]["applied_bits"][key] = 8
            elif edit == "bytes":
                trace[1]["kv"]["storage_bytes"] += 2
            elif edit == "cache":
                trace[1]["cache_id_in"] += 1
            else:
                trace[1]["kv_before_sha256"][0] = "wrong"
            with self.assertRaises(AssertionError):
                audit_tiny.inspect_trace(trace, cfg, row["prompt"]["token_ids"], [8, 4, 8])


if __name__ == "__main__":
    unittest.main()
