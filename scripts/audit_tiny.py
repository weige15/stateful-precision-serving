"""Independent CPU artifact audit. Imports NONE of the harness/analysis helpers.

Recomputes the primary pair, controls, full token comparisons, KV arithmetic, raw
hashes and blocked branch. Supports only tiny fixtures, not future main evidence.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

if not __debug__:
    raise RuntimeError("Audit requires unoptimized Python; -O/PYTHONOPTIMIZE disables assertions")


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def verify_files(root):
    expected = json.loads((root/"manifest.json").read_text())
    files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"}
    assert files == set(expected), "manifest file-set mismatch"
    for name, info in expected.items():
        assert (root/name).stat().st_size == info["bytes"] and digest(root/name) == info["sha256"], name


def metrics(a, b):
    assert a.ndim == b.ndim == 1 and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    def probs(values):
        unscaled = [math.exp(float(x)-float(max(values))) for x in values]
        s = math.fsum(unscaled)
        return [x/s for x in unscaled]
    p, q = probs(a), probs(b)
    av = sorted(range(len(a)), key=lambda i: (-float(a[i]), i))
    bv = sorted(range(len(b)), key=lambda i: (-float(b[i]), i))
    js = 0.
    for x, y in zip(p, q):
        m = (x+y)/2
        if x:
            js += x*math.log(x/m)/2
        if y:
            js += y*math.log(y/m)/2
    return {"max_abs_logit_diff": max(abs(float(x)-float(y)) for x, y in zip(a, b)),
            "total_variation": math.fsum(abs(x-y) for x, y in zip(p, q))/2, "js_nats": js,
            "top1_changed": av[0] != bv[0], "top1_a": av[0], "top1_b": bv[0],
            "topk_overlap": len(set(av[:10]).intersection(bv[:10]))/10, "k": 10}


def inspect_trace(trace, cfg, ids=None, bits=None):
    assert len({t["cache_id_in"] for t in trace}) == 1
    offset, flattened = 0, []
    for i, t in enumerate(trace):
        tokens = t["token_ids"]
        flattened += tokens
        assert t["cache_id_in"] == t["cache_id_out"]
        assert t["cache_length_before"] == offset
        assert t["token_sha256"] == hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
        offset += len(tokens)
        assert t["cache_length_after"] == offset
        requested = bits[i] if bits else t["requested_bits"]
        assert t["requested_bits"] == requested
        assert len(t["applied_bits"]) == cfg["num_hidden_layers"]*7
        assert set(t["applied_bits"].values()) == {requested}
        if ids is not None:
            assert t["prefix_checked"] and t["prefix_preserved"]
            if i:
                assert t["kv_before_sha256"] == trace[i-1]["kv_after_sha256"]
        ts = t["kv"]["tensors"]
        assert len(ts) == cfg["num_hidden_layers"]*2
        one_shape = [1, cfg["num_key_value_heads"], offset, cfg["head_dim"]]
        one_bytes = math.prod(one_shape)*2
        for j, tensor in enumerate(ts):
            assert tensor["shape"] == one_shape and tensor["dtype"] == "torch.float16"
            assert tensor["element_size"] == 2 and tensor["numel"] == math.prod(one_shape)
            assert tensor["layer"] == j//2 and tensor["kind"] == ("key" if j%2 == 0 else "value")
            assert tensor["tensor_bytes"] == tensor["storage_bytes"] == one_bytes
            assert tensor["storage_group"] == j  # actual DynamicCache dense independent allocations
        assert t["kv"]["tensor_bytes"] == t["kv"]["storage_bytes"] == one_bytes*len(ts)
    if ids is not None:
        assert flattened == ids, "teacher token mismatch"


def inspect_run(root):
    verify_files(root)
    result = json.loads((root/"result.json").read_text())
    assert result["stage"] == "tiny" and not result["valid_for_scientific_decision"]
    cfg = json.loads((root/"model.json").read_text())["config"]
    rows = []
    for file in result["prompt_files"]:
        p = json.loads((root/file).read_text())
        ids = p["prompt"]["token_ids"]
        assert len(ids) == 17
        arrays = {}
        for t in p["teacher"]:
            order = t["order"]
            inspect_trace(t["trace"], cfg, ids, [*order, 8])
            assert [len(e["token_ids"]) for e in t["trace"]] == [8, 8, 1]
            if order[0] != order[1]:
                counts = {b: sum(len(e["token_ids"]) for e in t["trace"][:-1] if e["requested_bits"] == b) for b in (4, 8)}
                assert counts[4] == counts[8] == 8
            arrays[(tuple(order), t["repeat"])] = np.load(root/t["file"], allow_pickle=False)
        assert len(arrays) == 8
        for order in ((8, 4), (4, 8), (8, 8), (4, 4)):
            assert np.array_equal(arrays[(order, 0)], arrays[(order, 1)])
        controls = []
        for c in p["controls"]:
            for trace in c["traces"]:
                inspect_trace(trace, cfg, ids, [c["bits"]]*3)
            one, chunk, repeat = [np.load(root/c["files"][k], allow_pickle=False) for k in ("oneshot", "chunked", "repeat")]
            assert one.shape == chunk.shape == repeat.shape == (1, 17, cfg["vocab_size"])
            delta = float(np.max(np.abs(one.astype(float)-chunk.astype(float))))
            delta_repeat = float(np.max(np.abs(chunk.astype(float)-repeat.astype(float))))
            assert delta <= 0.002 and delta_repeat == 0
            assert delta == c["chunk_max_abs"] and delta_repeat == c["repeat_max_abs"]
            controls.append({"bits": c["bits"], "chunk_max_abs": delta, "repeat_max_abs": delta_repeat})
        free = p["free"]
        assert set(free) == {"fixed4", "fixed6", "fixed8", "4to8", "8to4"}
        for name, f in free.items():
            assert f["prompt_token_ids"] == ids
            inspect_trace(f["trace"], cfg)
            decode = f["trace"][1:]
            length = len(f["token_ids"])
            assert length == len(decode) == f["output_length"] <= 8
            assert f["eos"] == (f["token_ids"][-1] == 63)
            assert f["length_cap_hit"] == (length == 8)
            assert f["right_censored"] == (length == 8 and not f["eos"])
            expected = ([int(name[-1])]*8 if name.startswith("fixed") else
                        [int(name[0])]*4+[int(name[-1])]*4)
            assert f["trace"][0]["token_ids"] == ids[:-1]
            assert f["trace"][0]["requested_bits"] == expected[0]
            assert [t["requested_bits"] for t in decode] == expected[:length]
            assert [t["output_token"] for t in decode] == f["token_ids"]
            assert [t["token_ids"][0] for t in decode] == [ids[-1]]+f["token_ids"][:-1]
            counts = {str(b): expected[:length].count(b) for b in set(expected[:length])}
            assert f["decode_exposure"] == counts
            total = dict(counts)
            total[str(expected[0])] += len(ids)-1
            assert total == f["total_exposure"]
            assert f["equal_decode_exposure"] == (counts.get("4", 0) > 0 and counts.get("4") == counts.get("8") and not counts.get("6"))
            assert f["equal_total_exposure"] == (total.get("4", 0) > 0 and total.get("4") == total.get("8") and not total.get("6"))
            sizes = [t["kv"]["storage_bytes"] for t in f["trace"]]
            assert f["peak_kv_storage_bytes"] == max(sizes)
            assert f["cumulative_kv_byte_forwards"] == sum(sizes)
        pairs = {}
        for name, a in free.items():
            if name == "fixed8":
                continue
            b = free["fixed8"]
            divergence = next((i for i, (x, y) in enumerate(itertools.zip_longest(a["token_ids"], b["token_ids"])) if x != y), None)
            pairs[name] = {"length_delta": len(a["token_ids"])-len(b["token_ids"]),
                           "abs_length_delta": abs(len(a["token_ids"])-len(b["token_ids"])),
                           "eos_delta": int(a["eos"])-int(b["eos"]), "eos_discordant": int(a["eos"] != b["eos"]),
                           "cap_delta": int(a["length_cap_hit"])-int(b["length_cap_hit"]), "first_divergence": divergence,
                           "peak_kv_delta": a["peak_kv_storage_bytes"]-b["peak_kv_storage_bytes"],
                           "cumulative_kv_delta": a["cumulative_kv_byte_forwards"]-b["cumulative_kv_byte_forwards"],
                           "abs_relative_cumulative_kv_delta": abs(a["cumulative_kv_byte_forwards"]-b["cumulative_kv_byte_forwards"])/b["cumulative_kv_byte_forwards"]}
        rows.append({"id": p["prompt"]["id"], "primary": metrics(arrays[((8, 4), 0)], arrays[((4, 8), 0)]),
                     "controls": controls, "fresh_cache_repeat_exact": True, "free_vs_fixed8": pairs})
    return rows


def near(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            near(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            near(x, y)
    elif isinstance(a, float):
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def validate_analysis(analysis, runs, rows):
    if len(analysis.get("runs", [])) != 2 or len(runs) != 2 or len(rows) != 2:
        raise ValueError("exactly two complete analysis runs required")
    for expected_path, independent, dependent in zip(runs, rows, analysis["runs"]):
        if Path(dependent["run"]).resolve() != expected_path.resolve():
            raise ValueError("analysis run identity mismatch")
        near(independent, dependent["rows"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = [inspect_run(p) for p in args.runs]  # compute BEFORE reading analysis
    arrays_a, arrays_b = [sorted(p.glob("*.npy")) for p in args.runs]
    assert [p.name for p in arrays_a] == [p.name for p in arrays_b]
    assert all(np.array_equal(np.load(a, allow_pickle=False), np.load(b, allow_pickle=False)) for a, b in zip(arrays_a, arrays_b))
    pids = [json.loads((p/"environment.json").read_text())["pid"] for p in args.runs]
    assert pids[0] != pids[1]
    analysis = json.loads(args.analysis.read_text())
    validate_analysis(analysis, args.runs, rows)
    assert rows[0] == rows[1]
    # Independently derive the branch from actual stage, not an analysis validity flag.
    stages = [json.loads((p/"result.json").read_text())["stage"] for p in args.runs]
    real_main_n = 0 if stages == ["tiny", "tiny"] else None
    assert real_main_n == 0
    decision = "pause_blocked"  # 0 < 16, no frozen real pilot/main processes
    assert analysis["decision"] == decision
    with (args.out/"audit.json").open("x") as f:
        json.dump({"passed": True, "scope": "tiny CPU fixtures ONLY; not completion of real study", "rows": rows,
                   "arrays_compared": len(arrays_a), "cpu_processes": pids, "main_prompts": real_main_n,
                   "decision": decision}, f, indent=2, sort_keys=True)
        f.write("\n")
    print("INDEPENDENT CPU AUDIT PASSED; pause_blocked (no real main evidence)")


if __name__ == "__main__":
    main()
