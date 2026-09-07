"""Analysis path. Audit recomputes these operations independently."""
import itertools
import math
import numpy as np


def distribution(logits):
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("finite vocabulary logits required")
    p = np.exp(x - x.max())
    return p / p.sum()


def logit_metrics(a, b, k=10):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or type(k) is not int or not 1 <= k <= a.size:
        raise ValueError("equal shapes and valid top-k required")
    p, q = distribution(a), distribution(b)
    # Stable sort: smaller token ID wins tied logits, as torch.argmax does.
    topa, topb = np.argsort(-a, kind="stable"), np.argsort(-b, kind="stable")
    m = (p+q)/2
    # xlogy convention avoids NaN for exact probability underflow.
    def kl(x, y):
        ix = x > 0
        return np.sum(x[ix] * (np.log(x[ix]) - np.log(y[ix])))
    return {"max_abs_logit_diff": float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))),
            "total_variation": float(np.abs(p-q).sum()/2), "js_nats": float((kl(p, m)+kl(q, m))/2),
            "top1_changed": bool(topa[0] != topb[0]), "top1_a": int(topa[0]), "top1_b": int(topb[0]),
            "topk_overlap": len(set(topa[:k]) & set(topb[:k]))/k, "k": k}


def first_divergence(a, b):
    for i, (x, y) in enumerate(itertools.zip_longest(a, b, fillvalue=None)):
        if x != y:
            return i  # 0-based; a length-only difference is at min(len(a),len(b))
    return None


def bootstrap_mean(values, seed=314159, draws=10000, confidence=0.95):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 < confidence < 1 or draws < 100:
        raise ValueError("invalid paired bootstrap input")
    indices = np.random.default_rng(seed).integers(0, len(x), size=(draws, len(x)))
    means = x[indices].mean(axis=1)
    alpha = (1-confidence)/2
    lo, hi = np.quantile(means, [alpha, 1-alpha], method="linear")
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": len(x),
            "confidence": confidence, "seed": seed, "draws": draws}


def paired_free(a, b):
    if a["prompt_token_ids"] != b["prompt_token_ids"]:
        raise ValueError("free-running prompts differ")
    return {"length_delta": a["output_length"]-b["output_length"],
            "abs_length_delta": abs(a["output_length"]-b["output_length"]),
            "eos_delta": int(a["eos"])-int(b["eos"]), "eos_discordant": int(a["eos"] != b["eos"]),
            "cap_delta": int(a["length_cap_hit"])-int(b["length_cap_hit"]),
            "first_divergence": first_divergence(a["token_ids"], b["token_ids"]),
            "peak_kv_delta": a["peak_kv_storage_bytes"]-b["peak_kv_storage_bytes"],
            "cumulative_kv_delta": a["cumulative_kv_byte_forwards"]-b["cumulative_kv_byte_forwards"],
            "abs_relative_cumulative_kv_delta": abs(a["cumulative_kv_byte_forwards"]-b["cumulative_kv_byte_forwards"])/b["cumulative_kv_byte_forwards"]}


def decide(*, n, frozen, valid, fresh_processes, repeat_agreement, history_pass, free_pass):
    """Missing/invalid/minimum evidence has priority over every scientific branch."""
    bools = (frozen, valid, repeat_agreement, history_pass, free_pass)
    if any(type(b) is not bool for b in bools) or type(n) is not int or type(fresh_processes) is not int:
        raise ValueError("explicit validity facts required")
    if n < 16 or not frozen or not valid or fresh_processes != 2 or not repeat_agreement:
        return "pause_blocked"
    if history_pass and free_pass:
        return "continue_joint_state_work"
    if history_pass:
        return "continue_precision_provenance"
    if free_pass:
        return "revise_endogenous_workload"
    return "stop_precision_state_direction"


def history_gate(metrics, noise_gate, practical_tv, **bootstrap):
    if not metrics or not all(math.isfinite(t) and t > 0 for t in (noise_gate, practical_tv)):
        raise ValueError("positive frozen thresholds required")
    noise = bootstrap_mean([m["max_abs_logit_diff"] for m in metrics], **bootstrap)
    practical = bootstrap_mean([m["total_variation"] for m in metrics], **bootstrap)
    return {"noise": noise, "practical": practical,
            "passes": noise["lo"] > noise_gate and practical["lo"] > practical_tv}


def free_gate(pairs, length_gate, eos_gate, kv_gate, **bootstrap):
    if not all(math.isfinite(t) and t > 0 for t in (length_gate, eos_gate, kv_gate)):
        raise ValueError("positive practical thresholds required")
    summaries = {k: bootstrap_mean([p[k] for p in pairs], **bootstrap) for k in
                 ("abs_length_delta", "eos_discordant", "abs_relative_cumulative_kv_delta")}
    return {"summaries": summaries, "passes": any(summaries[k]["lo"] > v for k, v in
            (("abs_length_delta", length_gate), ("eos_discordant", eos_gate), ("abs_relative_cumulative_kv_delta", kv_gate)))}
