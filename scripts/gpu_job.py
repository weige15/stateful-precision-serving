"""One serial GPU job, recorded process-free preflight, timeout and goal budget.

Never resets a GPU or signals existing processes. Timeout/interrupt cleanup signals
only the newly launched owned process group. No polling/wait-for-GPU loop.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
BUDGET_SECONDS = 4*60*60
TIMEOUT_SECONDS = 30*60
RESERVE_SECONDS = TIMEOUT_SECONDS+60  # conservative margin for process teardown


def dump(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def select_idle(xml):
    """Fail closed on unknown memory/utilization/process fields, including graphics."""
    candidates, rows = [], []
    root = ET.fromstring(xml)
    for index, g in enumerate(root.findall("gpu")):
        row = {"index": index, "uuid": g.findtext("uuid"), "name": g.findtext("product_name"),
               "free": g.findtext("fb_memory_usage/free"), "used": g.findtext("fb_memory_usage/used"),
               "util": g.findtext("utilization/gpu_util"),
               "processes": [p.findtext("pid") for p in g.findall("processes/process_info")]}
        rows.append(row)
        try:
            free = int(row["free"].removesuffix(" MiB"))
            used = int(row["used"].removesuffix(" MiB"))
            util = int(row["util"].removesuffix(" %"))
        except (AttributeError, ValueError):
            continue
        processes = g.find("processes")
        if (processes is not None and not (processes.text or "").strip() and not list(processes)
            and row["uuid"] and row["uuid"].startswith("GPU-") and free >= 22000 and used <= 1024 and util == 0):
            candidates.append(row["uuid"])
    return (candidates[0] if candidates else None), rows


def charged_seconds(root):
    total = 0
    for file in Path(root).glob("*/reservation.json"):
        reservation = json.loads(file.read_text())
        result = file.parent/"completion.json"
        charge = json.loads(result.read_text())["charged_seconds"] if result.exists() else reservation["reserved_seconds"]
        if not isinstance(charge, (int, float)) or not math.isfinite(charge) or charge < 0:
            raise ValueError("invalid GPU budget ledger")
        total += charge
    return total


def execute_owned(command, *, cwd, env, log, timeout_seconds=TIMEOUT_SECONDS):
    """Reap our process group on timeout/interrupt, never an unrelated process.

    Keep the group leader unreaped until the final kill so its PID cannot be
    recycled during cleanup. Block another SIGINT/SIGTERM during the bounded
    cleanup. A cleanup exception propagates: caller must NOT mark completion.
    """
    proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        code = proc.wait(timeout=max(0.01, timeout_seconds-5))
        return code, code in (124, 137)
    except BaseException as exc:
        old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM})
        try:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            time.sleep(0.2)  # bounded grace; do not poll/reap group leader before final kill
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=4)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
        if isinstance(exc, subprocess.TimeoutExpired):
            return 124, True
        if isinstance(exc, KeyboardInterrupt):
            return 130, False
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("pilot", "main"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    # Exactly one-level job dirs permit complete budget accounting with glob above.
    if out.parent != ROOT/"raw/gpu-jobs":
        parser.error("--out must be raw/gpu-jobs/NEW_JOB_ID")
    out.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out.parent/"serial.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        out.mkdir(exist_ok=False)
        dump(out/"command.json", {"argv": sys.argv, "executable": sys.executable, "cwd": str(ROOT),
                                 "utc": datetime.now(timezone.utc).isoformat(), "check_only": args.check_only})
        used = charged_seconds(out.parent)
        dump(out/"budget_before.json", {"charged_seconds": used, "budget_seconds": BUDGET_SECONDS,
                                       "next_reservation_seconds": RESERVE_SECONDS})
        selected = None
        for i in range(2):
            p = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=20)
            with (out/f"preflight-{i}.xml").open("x") as f:
                f.write(p.stdout)
            if p.returncode:
                dump(out/"blocked.json", {"reason": "nvidia-smi failed", "stderr": p.stderr, "gpu_seconds": 0})
                return 3
            uuid, rows = select_idle(p.stdout)
            dump(out/f"preflight-{i}.json", {"selected": uuid, "gpus": rows})
            if uuid is None or (selected is not None and selected != uuid):
                dump(out/"blocked.json", {"reason": "no stable idle process-free GPU with >=22000 MiB free", "gpu_seconds": 0})
                return 3
            selected = uuid
            if i == 0:
                time.sleep(1)
        if args.check_only:
            dump(out/"checked.json", {"selected": selected, "gpu_seconds": 0, "does_not_reserve_gpu": True})
            return 0
        pending = [str(p.parent) for p in out.parent.glob("*/reservation.json") if not (p.parent/"completion.json").exists()]
        if pending:
            dump(out/"blocked.json", {"reason": "unfinished prior job; verify owned child stopped before recovery", "pending": pending, "gpu_seconds": 0})
            return 3
        if used + RESERVE_SECONDS > BUDGET_SECONDS:
            dump(out/"blocked.json", {"reason": "four GPU-hour budget would be exceeded", "gpu_seconds": 0})
            return 3
        # Import CPU-only authorization code only after preflight; no CUDA init here.
        sys.path[:0] = [str(ROOT/"src"), str(ROOT/"vendor")]
        from precision_state.artifacts import sha256
        from precision_state.protocol import authorize_main
        config_path = ROOT/"configs/precision_state_v1.json"
        config = json.loads(config_path.read_text())
        if args.stage == "main":
            authorize_main(config, ROOT)
            mains = [f for f in out.parent.glob("*/reservation.json") if json.loads(f.read_text())["stage"] == "main"]
            if len(mains) >= 2:
                raise ValueError("two fresh main attempts already reserved; no silent replacements")
        else:
            pilots = list(out.parent.glob("*/reservation.json"))
            if any(json.loads(f.read_text())["stage"] == "pilot" for f in pilots):
                raise ValueError("pilot already attempted; new authorization required to retry")
        reservation = {"reserved_seconds": RESERVE_SECONDS, "timeout_seconds": TIMEOUT_SECONDS, "gpu_uuid": selected,
                       "stage": args.stage, "config_sha256": sha256(config_path), "output": str(out/"data")}
        dump(out/"reservation.json", reservation)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": selected, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
               "PRECISION_STATE_GPU_GUARD": "1", "PRECISION_STATE_JOB_AUTH": str(out/"reservation.json"),
               "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TOKENIZERS_PARALLELISM": "false",
               "PYTHONPATH": f"{ROOT}/src:{ROOT}/vendor"}
        # Child-owned timeout survives wrapper death; its process group contains only
        # this new job. The outer Python timeout is a second safety bound.
        command = ["timeout", "--signal=TERM", "--kill-after=5s", "1790s", sys.executable,
                   str(ROOT/"scripts/run_study.py"), "--stage", args.stage, "--out", str(out/"data")]
        dump(out/"launch.json", {"command": command, "gpu_uuid": selected})
        start = time.monotonic()
        try:
            with (out/"stdout.log").open("x") as log:
                code, timeout = execute_owned(command, cwd=ROOT, env=env, log=log)
        except BaseException as exc:
            # Fail closed: an unknown cleanup state leaves reservation unfinished.
            dump(out/"interrupted.json", {"error": repr(exc), "wall_seconds": time.monotonic()-start})
            raise
        dump(out/"completion.json", {"returncode": code, "timeout": timeout,
                                    "owned_job_stopped": True, "charged_seconds": time.monotonic()-start})
        return code


if __name__ == "__main__":
    sys.exit(main())
