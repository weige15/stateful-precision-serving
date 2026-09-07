"""Analysis of CPU correctness fixtures only; NEVER a Qwen3-4B decision path."""
import argparse
import json
from pathlib import Path
import numpy as np
from precision_state.artifacts import new_run, seal, verify, write_json
from precision_state.analysis import decide, logit_metrics, paired_free, bootstrap_mean


def analyze(root):
    verify(root)
    result = json.loads((root/"result.json").read_text())
    if result["stage"] != "tiny":
        raise ValueError("this analysis is only for tiny CPU correctness artifacts")
    rows = []
    for file in result["prompt_files"]:
        p = json.loads((root/file).read_text())
        teacher = {("".join(map(str, t["order"])), t["repeat"]): t for t in p["teacher"]}
        arrays = {k: np.load(root/t["file"], allow_pickle=False) for k, t in teacher.items()}
        repeat = all(np.array_equal(arrays[(order, 0)], arrays[(order, 1)]) for order in ("84", "48", "88", "44"))
        controls = [{"bits": c["bits"], "chunk_max_abs": float(np.max(np.abs(np.load(root/c["files"]["oneshot"])-np.load(root/c["files"]["chunked"])))),
                     "repeat_max_abs": float(np.max(np.abs(np.load(root/c["files"]["chunked"])-np.load(root/c["files"]["repeat"]))))} for c in p["controls"]]
        rows.append({"id": p["prompt"]["id"], "primary": logit_metrics(arrays[("84", 0)], arrays[("48", 0)]),
                     "controls": controls, "fresh_cache_repeat_exact": repeat,
                     "free_vs_fixed8": {name: paired_free(value, p["free"]["fixed8"]) for name, value in p["free"].items() if name != "fixed8"}})
    return {"run": str(root), "kind": "CPU_CORRECTNESS_ONLY", "rows": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = new_run(args.out)
    analyses = [analyze(r) for r in args.runs]
    agreement = analyses[0]["rows"] == analyses[1]["rows"]
    # Also compare every saved array, not just aggregate statistics.
    arrays = sorted(args.runs[0].glob("*.npy"))
    array_agreement = all(np.array_equal(np.load(p, allow_pickle=False), np.load(args.runs[1]/p.name, allow_pickle=False)) for p in arrays)
    envs = [json.loads((r/"environment.json").read_text()) for r in args.runs]
    facts = dict(n=0, frozen=False, valid=False, fresh_processes=0, repeat_agreement=False, history_pass=False, free_pass=False)
    write_json(out/"analysis.json", {"runs": analyses, "all_arrays_repeat_exact": array_agreement,
             "summary_repeat_exact": agreement, "distinct_cpu_pids": envs[0]["pid"] != envs[1]["pid"],
             "real_main_evidence": facts, "decision": decide(**facts),
             "reason": "No frozen real-model pilot/main evidence; tiny results cannot select a scientific branch."})
    seal(out)
    print("CPU analysis complete; scientific branch: pause_blocked")


if __name__ == "__main__":
    main()
