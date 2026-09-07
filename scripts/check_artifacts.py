"""Verify the preserved CPU/pause packet, NOT a scientific completion gate."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PIN = "a5f4d5d52358c7b9740ecec448db5279c6af5ddc"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    receipt = json.loads((ROOT/"evidence/artifacts.json").read_text())
    for name, expected in receipt["files"].items():
        p = ROOT/name
        require(p.is_file() and p.stat().st_size == expected["bytes"] and digest(p) == expected["sha256"], f"packet changed/missing: {p}")
    intake = json.loads((ROOT/"evidence/intake.json").read_text())
    require(intake["upstream_commit"] == PIN, "wrong source pin")
    for name, expected in intake["inputs"].items():
        require(digest(name) == expected["sha256"], f"imported input changed: {name}")
        if "expected_sha256" in expected:
            require(expected["sha256"] == expected["expected_sha256"], f"upstream evidence hash mismatch: {name}")
    for source, expected in intake["sources"].items():
        blob = subprocess.check_output(["git", "-C", str(ROOT/"inputs/qaq-pinned"), "show", f"{PIN}:{source}"])
        require(hashlib.sha256(blob).hexdigest() == expected["sha256"], f"pinned blob mismatch: {source}")
        if "imported_as" in expected:
            require(digest(ROOT/expected["imported_as"]) == expected["sha256"], f"vendor changed: {source}")
    from precision_state.protocol import select_prompts, authorize_main
    from precision_state.analysis import decide
    config = json.loads((ROOT/"configs/precision_state_v1.json").read_text())
    examples = [json.loads(s) for s in Path(config["input_examples"]).read_text().splitlines()]
    require(config["prompts"] == select_prompts(examples), "outcome-blind prompt selection mismatch")
    require(not (ROOT/"evidence/freeze.json").exists(), "packet describes unfrozen study but freeze exists")
    try:
        authorize_main(config, ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("main was unexpectedly authorized")
    jobs = list((ROOT/"raw/gpu-jobs").glob("*/reservation.json"))
    require(not jobs, "packet claims no GPU jobs but reservations exist")
    require(not list((ROOT/"raw/gpu-jobs").glob("*/data/result.json")), "unaccounted real study results")
    require("\n`pause_blocked`\n" in (ROOT/"DECISION.md").read_text(), "decision mismatch")
    analysis = json.loads((ROOT/"raw/analysis-001/data/analysis.json").read_text())
    audit = json.loads((ROOT/"raw/audit-002/data/audit.json").read_text())
    require(audit["main_prompts"] == 0 and audit["passed"], "CPU audit failed/misclassified")
    require(analysis["all_arrays_repeat_exact"] and analysis["summary_repeat_exact"] and analysis["distinct_cpu_pids"], "CPU repeat mismatch")
    actual = decide(n=0, frozen=False, valid=False, fresh_processes=0,
                    repeat_agreement=False, history_pass=False, free_pass=False)
    require(actual == analysis["decision"] == audit["decision"] == "pause_blocked", "branch mismatch")
    # Compare selected upstream files/status to the initial read-only inventory.
    initial = next(c["stdout"] for c in intake["commands"] if c["argv"][-2:] == ["status", "--short"])
    import os
    status = subprocess.check_output(["git", "-C", str(ROOT.parent/"qaq_baseline"), "status", "--short"], text=True,
                                     env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    require(status == initial, "upstream worktree status changed; inspect without repairing")
    print(json.dumps({"packet_hashes_pass": len(receipt["files"]), "pinned_blobs_pass": len(intake["sources"]),
                      "input_hashes_pass": len(intake["inputs"]), "upstream_status_unchanged": True,
                      "main_prompts": 0, "gpu_seconds": 0, "decision": actual,
                      "goal_complete": False, "scope": "CPU correctness and blockage only"}, indent=2))


if __name__ == "__main__":
    main()
