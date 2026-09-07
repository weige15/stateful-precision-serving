"""Serial CPU correctness / guarded real pilot / frozen real main runner.

No model is downloaded, no evidence is regenerated, no package is changed.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import torch

from precision_state.artifacts import environment, new_run, save_array, seal, sha256, write_json
from precision_state.harness import CachedRunner, free_schedules, tiny_model, token_hash
from precision_state.analysis import logit_metrics
from precision_state.protocol import authorize_main

ROOT = Path(__file__).resolve().parents[1]


def persist_control(out, name, runner, ids, bits, chunk):
    started = time.monotonic()
    arrays, traces = runner.constant_control(ids, bits, chunk)
    files = {}
    for key, value in arrays.items():
        file = f"{name}-{key}.npy"
        save_array(out/file, value.numpy())
        files[key] = file
    return {"bits": bits, "files": files, "traces": traces,
            "chunk_max_abs": float((arrays["oneshot"]-arrays["chunked"]).abs().max()),
            "repeat_max_abs": float((arrays["chunked"]-arrays["repeat"]).abs().max()),
            "seconds": time.monotonic()-started}


def run_workload(out, runner, prompts, chunk, segment, eos_ids, stage, tolerance):
    records = []
    # Controls for ALL prompts precede ANY cross-schedule result.
    for i, p in enumerate(prompts):
        control = [persist_control(out, f"p{i}-control{b}", runner, p["token_ids"], b, chunk) for b in (8, 4)]
        if any(c["chunk_max_abs"] > tolerance or c["repeat_max_abs"] != 0 for c in control):
            write_json(out/f"failed-control-p{i}.json", {"prompt": p, "controls": control})
            raise ValueError("constant precision chunk/repeat controls failed; no interpretation")
        records.append({"prompt": p, "controls": control, "teacher": [], "free": {}})
    for i, row in enumerate(records):
        ids = row["prompt"]["token_ids"]
        if stage != "pilot":
            for order in ([8, 4], [4, 8], [8, 8], [4, 4]):
                for repeat in range(2):
                    logits, trace = runner.teacher(ids, order, chunk)
                    file = f"p{i}-teacher{order[0]}{order[1]}-r{repeat}.npy"
                    save_array(out/file, logits.numpy())
                    row["teacher"].append({"order": order, "repeat": repeat, "file": file, "trace": trace})
        schedules = free_schedules(segment)
        if stage == "pilot":
            schedules = {k: v for k, v in schedules.items() if k.startswith("fixed")}
        for name, schedule in schedules.items():
            started = time.monotonic()
            row["free"][name] = runner.generate(ids, schedule, eos_ids)
            row["free"][name]["seconds"] = time.monotonic()-started
        write_json(out/f"prompt-{i}.json", row)
        print(f"completed {stage} prompt {i+1}/{len(records)}", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["tiny", "pilot", "main"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.out.resolve().is_relative_to(ROOT/"raw"):
        parser.error("all outputs must be under this repository's raw/")
    # Fail authorization before GPU initialization or model deserialization.
    config = None
    if args.stage != "tiny":
        config = json.loads((ROOT/"configs/precision_state_v1.json").read_text())
        if args.stage == "main":
            authorize_main(config, ROOT)
        if os.environ.get("PRECISION_STATE_GPU_GUARD") != "1" or not os.environ.get("CUDA_VISIBLE_DEVICES"):
            parser.error("real runs require scripts/gpu_job.py preflight")
        authorization = json.loads(Path(os.environ["PRECISION_STATE_JOB_AUTH"]).read_text())
        if (authorization["stage"] != args.stage or authorization["config_sha256"] != sha256(ROOT/"configs/precision_state_v1.json")
                or authorization["output"] != str(args.out.resolve())):
            parser.error("job authorization mismatch")
    out = new_run(args.out)
    start = time.monotonic()
    try:
        torch.set_num_threads(1 if args.stage == "tiny" else 4)
        torch.manual_seed(1729)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        write_json(out/"environment.json", environment())
        source_files = [p for d in ("src", "vendor", "scripts", "configs", "tests") for p in (ROOT/d).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
        source_map = {}
        for p in source_files:
            rel = p.relative_to(ROOT)
            dest = out/"source"/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("xb") as f:
                f.write(p.read_bytes())
            source_map[str(rel)] = sha256(dest)
        write_json(out/"source_manifest.json", source_map)
        if args.stage == "tiny":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
                raise ValueError("CPU run must explicitly hide GPUs")
            model = tiny_model()
            chunk, segment, eos = 8, 4, [63]
            prompts = [{"id": f"synthetic:{i}", "token_ids": [(j*7+i*11)%62+1 for j in range(17)]} for i in range(2)]
            for p in prompts:
                p["token_sha256"] = token_hash(p["token_ids"])
            tolerance = 0.002
            model_info = {"kind": "random_tiny_Qwen3_FP16_CPU_NOT_Qwen3-4B", "seed": 1729, "config": model.config.to_dict()}
        else:
            from qaq.model import load_quantized
            # Verify ALL imported model/tokenizer/input sources before load.
            intake = json.loads((ROOT/config["intake_path"]).read_text())
            if sha256(ROOT/config["intake_path"]) != config["intake_sha256"]:
                raise ValueError("intake hash mismatch")
            for p, expected in intake["inputs"].items():
                if sha256(p) != expected["sha256"]:
                    raise ValueError(f"upstream input changed: {p}")
            for p, expected in intake["sources"].items():
                if "imported_as" in expected and sha256(ROOT/expected["imported_as"]) != expected["sha256"]:
                    raise ValueError(f"vendor changed: {p}")
            # Hashing/model provenance may take time. Recheck process-free status
            # immediately before touching CUDA; do not rely on the older wrapper sample.
            import subprocess
            from gpu_job import select_idle
            preflight = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True, timeout=20)
            with (out/"pre_cuda.xml").open("x") as f:
                f.write(preflight.stdout)
            if preflight.returncode or select_idle(preflight.stdout)[0] != os.environ["CUDA_VISIBLE_DEVICES"]:
                raise ValueError("GPU no longer idle/process-free before CUDA initialization")
            if torch.cuda.device_count() != 1:
                raise ValueError("exactly one visible GPU required")
            model, metadata = load_quantized(config["checkpoint"], "cuda:0")
            chunk, segment, eos = config["chunk_tokens"], config["decode_segment_tokens"], config["eos_ids"]
            prompts = config["prompts"][args.stage]
            if args.stage == "pilot" and not 1 <= len(prompts) <= 4:
                raise ValueError("pilot cap violated")
            tolerance = config["pilot_control_ceiling"] if args.stage == "pilot" else config["constant_control_tolerance"]
            model_info = {"kind": "Qwen3-4B_nested_FP16_reconstruction_NOT_low_bit_kernel", "metadata": metadata,
                          "gpu": torch.cuda.get_device_name(0), "intake_sha256": config["intake_sha256"]}
        write_json(out/"model.json", model_info)
        load_seconds = time.monotonic()-start
        rows = run_workload(out, CachedRunner(model), prompts, chunk, segment, eos, args.stage, tolerance)
        elapsed = time.monotonic()-start
        write_json(out/"result.json", {"stage": args.stage, "valid_for_scientific_decision": args.stage == "main",
                    "prompt_ids": [r["prompt"]["id"] for r in rows], "controls_passed": True,
                    "load_seconds": load_seconds, "wall_seconds": elapsed,
                    "constant_control_tolerance": tolerance, "prompt_files": [f"prompt-{i}.json" for i in range(len(rows))],
                    "config_sha256": sha256(ROOT/"configs/precision_state_v1.json") if config else None})
        print(f"{args.stage} complete: {elapsed:.3f}s (not a decision without independent audit)", flush=True)
    except BaseException:
        write_json(out/"failure.json", {"traceback": traceback.format_exc(), "wall_seconds": time.monotonic()-start})
        raise
    finally:
        seal(out)


if __name__ == "__main__":
    main()
