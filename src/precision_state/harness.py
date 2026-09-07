"""Cached forwards with observed (not just requested) precision and KV accounting.

The imported NestedLinear reconstructs float weights for torch linear, NOT a low-bit
kernel. DynamicCache is neither quantized nor rebuilt when weight precision changes.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json

import torch
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM
from qaq.model import NestedLinear, blocks, install_replacements, prepare_replacements, set_profile


@dataclass(frozen=True)
class Segment:
    tokens: int
    bits: int


def validate_schedule(segments, total, *, balanced=False, probe=False):
    if type(total) is not int or total <= 0 or not segments:
        raise ValueError("nonempty positive schedule required")
    for s in segments:
        if type(s.tokens) is not int or s.tokens <= 0 or type(s.bits) is not int or s.bits not in (4, 6, 8):
            raise ValueError("invalid segment length/precision")
    if sum(s.tokens for s in segments) != total:
        raise ValueError("schedule must cover tokens exactly")
    if probe and (len(segments) != 3 or segments[-1] != Segment(1, 8)):
        raise ValueError("teacher probe must be exactly one W8 token after two chunks")
    history = segments[:-1] if probe else segments
    counts = Counter()
    for s in history:
        counts[s.bits] += s.tokens
    if balanced and (counts[4] == 0 or counts[4] != counts[8] or counts[6] != 0):
        raise ValueError("history must contain equal positive W4/W8 counts only")
    if probe and history[0].tokens != history[1].tokens:
        raise ValueError("history chunks must have equal lengths")
    return dict(counts)


def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def cache_tensors(cache):
    if cache is None:
        return []
    if not isinstance(cache, DynamicCache):
        raise TypeError("this study only supports the tested DynamicCache API")
    return [(layer.keys, layer.values) for layer in cache.layers if layer.keys is not None]


def kv_metadata(cache):
    """Exact live tensor bytes and deduplicated backing storage after a forward.

    Storage bytes include unused view backing. They exclude CUDA allocator rounding,
    temporary concat tensors, weights and activations. Cumulative bytes later means
    sum of these snapshots (byte-forwards), never allocation/traffic volume.
    """
    rows, storages = [], {}
    for index, pair in enumerate(cache_tensors(cache)):
        for kind, t in zip(("key", "value"), pair):
            storage = t.untyped_storage()
            identity = (str(t.device), storage.data_ptr())
            storages[identity] = storage.nbytes()
            rows.append({"layer": index, "kind": kind, "shape": list(t.shape),
                         "dtype": str(t.dtype), "device": str(t.device), "stride": list(t.stride()),
                         "offset": t.storage_offset(), "numel": t.numel(),
                         "element_size": t.element_size(), "tensor_bytes": t.numel() * t.element_size(),
                         "storage_bytes": storage.nbytes(), "storage_group": list(storages).index(identity)})
    return {"tensors": rows, "tensor_bytes": sum(r["tensor_bytes"] for r in rows),
            "storage_bytes": sum(storages.values())}


def cache_hashes(cache):
    return [hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
            for pair in cache_tensors(cache) for t in pair]


def tiny_model(seed=1729):
    torch.manual_seed(seed)
    config = Qwen3Config(hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                        vocab_size=64, max_position_embeddings=1024, tie_word_embeddings=True)
    config._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(config).half().eval().requires_grad_(False)
    install_replacements(prepare_replacements(model, group_size=128))
    return model


class CachedRunner:
    def __init__(self, model):
        if model.training:
            raise ValueError("eval model required")
        self.model = model
        self.nblocks = len(list(blocks(model)))
        self.modules = {n: m for n, m in model.named_modules() if isinstance(m, NestedLinear)}
        if len(self.modules) != self.nblocks // 2 * 7:
            raise ValueError("every attention/MLP projection must be nested")
        self.device = next(model.parameters()).device

    @torch.inference_mode()
    def forward(self, ids, bits, cache, *, check_prefix=True):
        if (type(bits) is not int or bits not in (4, 6, 8) or not ids or
            any(type(i) is not int or not 0 <= i < self.model.config.vocab_size for i in ids)):
            raise ValueError("invalid bits/token IDs")
        before_len = cache.get_seq_length()
        old = [(k.clone(), v.clone()) for k, v in cache_tensors(cache)] if check_prefix else []
        before_hashes = cache_hashes(cache) if check_prefix else None
        set_profile(self.model, [bits] * self.nblocks)
        observed = {}
        def observe(name):
            def hook(module, _inputs):
                if name in observed:
                    raise AssertionError("projection called more than once in one forward")
                observed[name] = module.bits
            return hook
        hooks = [m.register_forward_pre_hook(observe(n)) for n, m in self.modules.items()]
        try:
            x = torch.tensor([ids], device=self.device, dtype=torch.long)
            output = self.model(x, past_key_values=cache, use_cache=True,
                                attention_mask=torch.ones((1, before_len + len(ids)), device=self.device, dtype=torch.long),
                                position_ids=torch.arange(before_len, before_len + len(ids), device=self.device)[None, :])
        finally:
            for h in hooks:
                h.remove()
        if output.past_key_values is not cache:
            raise AssertionError("model replaced the passed cache object")
        if len(observed) != len(self.modules) or any(b != bits for b in observed.values()):
            raise AssertionError("applied schedule disagrees with requested schedule")
        if cache.get_seq_length() != before_len + len(ids):
            raise AssertionError("cache did not append exactly the supplied tokens")
        tensors = cache_tensors(cache)
        if len(tensors) != self.nblocks // 2:
            raise AssertionError("missing layer caches")
        for pair in tensors:
            if any(t.shape[-2] != before_len + len(ids) for t in pair):
                raise AssertionError("per-layer cache length mismatch")
        if check_prefix:
            for previous, current in zip(old, tensors):
                for a, b in zip(previous, current):
                    if not torch.equal(a, b[..., :before_len, :]):
                        raise AssertionError("old KV prefix changed; cache reuse is invalid")
        logits = output.logits.detach().float().cpu()
        if not torch.isfinite(logits).all():
            raise ValueError("nonfinite logits")
        trace = {"requested_bits": bits, "applied_bits": observed, "token_ids": list(ids),
                 "token_sha256": token_hash(ids), "cache_id_in": id(cache), "cache_id_out": id(output.past_key_values),
                 "cache_length_before": int(before_len), "cache_length_after": int(cache.get_seq_length()),
                 "prefix_checked": check_prefix, "prefix_preserved": True if check_prefix else None,
                 "kv_before_sha256": before_hashes, "kv_after_sha256": cache_hashes(cache) if check_prefix else None,
                 "kv": kv_metadata(cache)}
        return logits, trace

    def sequence(self, ids, segments):
        validate_schedule(segments, len(ids))  # atomic: fail before model/cache mutation
        cache = DynamicCache(config=self.model.config)
        logits, trace, offset = [], [], 0
        for s in segments:
            chunk, event = self.forward(ids[offset:offset+s.tokens], s.bits, cache)
            logits.append(chunk)
            trace.append(event)
            offset += s.tokens
        return torch.cat(logits, dim=1), trace

    def teacher(self, ids, order, chunk):
        if len(order) != 2 or any(type(b) is not int or b not in (4, 8) for b in order):
            raise ValueError("teacher histories support two W4/W8 chunks")
        segments = [Segment(chunk, order[0]), Segment(chunk, order[1]), Segment(1, 8)]
        validate_schedule(segments, len(ids), balanced=order[0] != order[1], probe=True)
        logits, trace = self.sequence(ids, segments)
        return logits[0, -1], trace

    @torch.inference_mode()
    def constant_control(self, ids, bits, chunk):
        # Includes the probe under the same constant precision, unlike mixed/W8-probe study.
        if len(ids) != 2 * chunk + 1 or bits not in (4, 8):
            raise ValueError("invalid constant control")
        set_profile(self.model, [bits] * self.nblocks)
        x = torch.tensor([ids], dtype=torch.long, device=self.device)
        one = self.model(x, use_cache=False).logits.detach().float().cpu()
        a, trace_a = self.sequence(ids, [Segment(chunk, bits), Segment(chunk, bits), Segment(1, bits)])
        b, trace_b = self.sequence(ids, [Segment(chunk, bits), Segment(chunk, bits), Segment(1, bits)])
        return {"oneshot": one, "chunked": a, "repeat": b}, [trace_a, trace_b]

    def generate(self, ids, segments, eos_ids):
        """Greedy single-sample decoding. Prompt prefill uses first segment's bits.

        The two equal segments count decode input tokens, excluding prefill. Actual
        exposure includes prefill and is recorded; mixed runs are not equal total-bit
        budget comparisons. Final sampled/EOS token is NOT yet in the cache.
        """
        cap = sum(s.tokens for s in segments)
        validate_schedule(segments, cap)
        if len(ids) < 2 or not eos_ids or any(type(i) is not int or not 0 <= i < self.model.config.vocab_size for i in eos_ids):
            raise ValueError("prompt >=2 tokens and valid EOS IDs required")
        cache = DynamicCache(config=self.model.config)
        _, prefill = self.forward(ids[:-1], segments[0].bits, cache, check_prefix=False)
        output, trace, pending = [], [dict(prefill, phase="prefill")], [ids[-1]]
        decode_counts, all_counts = Counter(), Counter({segments[0].bits: len(ids)-1})
        stop = False
        for s in segments:
            for _ in range(s.tokens):
                logits, event = self.forward(pending, s.bits, cache, check_prefix=False)
                token = int(logits[0, -1].argmax().item())  # lowest token ID breaks exact ties
                output.append(token)
                event.update(phase="decode", output_token=token, output_index=len(output)-1)
                trace.append(event)
                pending = [token]
                decode_counts[s.bits] += 1
                all_counts[s.bits] += 1
                if token in eos_ids:
                    stop = True
                    break
            if stop:
                break
        values = [e["kv"]["storage_bytes"] for e in trace]
        return {"prompt_token_ids": list(ids), "token_ids": output, "eos": stop,
                "eos_token": output[-1] if stop else None, "output_length": len(output),
                "length_cap": cap, "length_cap_hit": len(output) == cap,
                "right_censored": not stop and len(output) == cap,
                "decode_exposure": dict(decode_counts), "total_exposure": dict(all_counts),
                "equal_decode_exposure": decode_counts[4] > 0 and decode_counts[4] == decode_counts[8] and not decode_counts[6],
                "equal_total_exposure": all_counts[4] > 0 and all_counts[4] == all_counts[8] and not all_counts[6],
                "peak_kv_storage_bytes": max(values), "cumulative_kv_byte_forwards": sum(values),
                "trace": trace}


def free_schedules(segment):
    if type(segment) is not int or segment <= 0:
        raise ValueError("positive segment length required")
    return {"fixed4": [Segment(2*segment, 4)], "fixed6": [Segment(2*segment, 6)],
            "fixed8": [Segment(2*segment, 8)], "4to8": [Segment(segment, 4), Segment(segment, 8)],
            "8to4": [Segment(segment, 8), Segment(segment, 4)]}
