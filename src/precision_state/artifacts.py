"""Small append-only artifact helpers; no overwrite/reuse of result directories."""
import hashlib
import json
from pathlib import Path
import sys
import os
from datetime import datetime, timezone

import numpy as np


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")


def save_array(path, value):
    with Path(path).open("xb") as f:
        np.save(f, np.asarray(value), allow_pickle=False)


def new_run(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def manifest(path):
    path = Path(path)
    return {str(p.relative_to(path)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(path.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def seal(path):
    data = manifest(path)
    write_json(Path(path) / "manifest.json", data)
    return data


def verify(path):
    path = Path(path)
    expected = json.loads((path / "manifest.json").read_text())
    if expected != manifest(path):
        raise ValueError(f"artifact manifest mismatch: {path}")
    return expected


def environment():
    import torch
    import transformers
    return {"utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv, "cwd": os.getcwd(),
            "executable": sys.executable, "python": sys.version, "pid": os.getpid(),
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONPATH")}}
