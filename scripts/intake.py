"""Read-only upstream acquisition/provenance; no model downloads or regeneration."""
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone

PIN = "a5f4d5d52358c7b9740ecec448db5279c6af5ddc"
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT.parent / "qaq_baseline"


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def command(argv):
    p = subprocess.run(argv, text=True, capture_output=True, timeout=30,
                       env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"})
    return {"argv": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main():
    out = ROOT / sys.argv[1]
    out.mkdir(parents=True, exist_ok=False)
    git = ["git", "-C", str(ROOT / "inputs/qaq-pinned")]
    commit = subprocess.check_output([*git, "rev-parse", "FETCH_HEAD"], text=True).strip()
    if commit != PIN:
        raise ValueError("source pin mismatch")
    # Hash every tree blob, even documentation not imported into Python.
    sources = {}
    paths = subprocess.check_output([*git, "ls-tree", "-r", "--name-only", PIN], text=True).splitlines()
    for path in paths:
        blob = subprocess.check_output([*git, "show", f"{PIN}:{path}"])
        sources[path] = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)}
        if path in ("src/qaq/model.py", "src/qaq/quantization.py"):
            dest = ROOT / "vendor" / path.removeprefix("src/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                if digest(dest) != sources[path]["sha256"]:
                    raise ValueError(f"modified vendor: {dest}")
            else:
                with dest.open("xb") as f:
                    f.write(blob)
            sources[path]["imported_as"] = str(dest.relative_to(ROOT))
    inputs = {}
    names = ["integration/quantized_model.pt", "integration/gate.json", "integration/state_hashes.json",
             "frozen/model_manifest.json", "frozen/examples.jsonl", "frozen/data_manifest.json",
             "frozen/protocol.json", "frozen/freeze_hashes.json"]
    for name in names:
        path = UPSTREAM / "results/core-v1" / name
        inputs[str(path)] = {"exists": path.is_file()}
        if path.is_file():
            inputs[str(path)].update(sha256=digest(path), bytes=path.stat().st_size)
    info = json.loads((UPSTREAM / "results/core-v1/frozen/model_manifest.json").read_text())
    for name, expected in info["file_sha256"].items():
        path = Path(info["local_path"]) / name
        actual = digest(path) if path.is_file() else None
        inputs[str(path)] = {"exists": path.is_file(), "sha256": actual,
                             "expected_sha256": expected, "matches": actual == expected,
                             "bytes": path.stat().st_size if path.is_file() else None}
    checkpoint = UPSTREAM / "results/core-v1/integration/quantized_model.pt"
    gate = json.loads((checkpoint.parent / "gate.json").read_text())
    inputs[str(checkpoint)]["expected_sha256"] = gate["checkpoint_sha256"]
    inputs[str(checkpoint)]["matches"] = inputs[str(checkpoint)]["sha256"] == gate["checkpoint_sha256"]
    import torch
    import transformers
    import numpy
    report = {"utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv,
              "cwd": str(ROOT), "executable": sys.executable, "python": sys.version,
              "platform": platform.platform(), "versions": {"torch": torch.__version__,
              "transformers": transformers.__version__, "numpy": numpy.__version__},
              "upstream_commit": PIN, "sources": sources, "inputs": inputs,
              "commands": [command(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"]),
                           command(["git", "-C", str(UPSTREAM), "status", "--short"]),
                           command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv"]),
                           command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"]),
                           command([sys.executable, "-m", "pip", "freeze"])]}
    report["accessible_and_hashes_valid"] = all(x["exists"] and x.get("matches", True) for x in inputs.values())
    with (out / "intake.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps({"out": str(out), "sha256": digest(out / "intake.json"),
                      "accessible_and_hashes_valid": report["accessible_and_hashes_valid"]}, indent=2))


if __name__ == "__main__":
    main()
