"""Nested signed weight quantization; a replication choice, not the paper's format.

One two's-complement int8 code and FP32 scale per input-column group. At lower
widths only the retained high bits determine the bin midpoint. In particular,
zero codes in NONZERO groups are not special-cased using discarded bits.
"""
from __future__ import annotations

import torch

PRECISIONS = (4, 6, 8)


def quantize(weight: torch.Tensor, group_size: int = 128):
    if weight.ndim != 2 or group_size <= 0 or weight.shape[1] % group_size:
        raise ValueError("expected a matrix with input columns divisible by group_size")
    if weight.numel() == 0 or not torch.isfinite(weight).all():
        raise ValueError("weights must be nonempty and finite")
    grouped = weight.float().reshape(weight.shape[0], -1, group_size)
    scale = grouped.abs().amax(-1, keepdim=True) / 127
    safe_scale = torch.where(scale == 0, 1, scale)
    q = (grouped / safe_scale).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def reconstruct(q: torch.Tensor, scale: torch.Tensor, bits: int,
                dtype: torch.dtype = torch.float16) -> torch.Tensor:
    if bits not in PRECISIONS:
        raise ValueError(f"bits must be one of {PRECISIONS}")
    if q.dtype != torch.int8 or q.ndim != 3 or scale.shape != q.shape[:-1] + (1,):
        raise ValueError("invalid grouped int8 code/scale shapes")
    shift = 8 - bits
    # Signed arithmetic shift keeps the sign bit and top (bits-1) magnitude bits
    # in two's complement; not sign-magnitude. Bin midpoints minimize uniform MSE.
    codes = (q.to(torch.int16) >> shift).float() * (1 << shift)
    codes += ((1 << shift) - 1) / 2
    return (codes * scale).reshape(q.shape[0], -1).to(dtype)


def block_linears(model):
    """The seven projection weights per Qwen3 layer; head/norms are excluded."""
    for index, layer in enumerate(model.model.layers):
        for kind, block in (("attn", layer.self_attn), ("ffn", layer.mlp)):
            for name, module in block.named_modules():
                if isinstance(module, torch.nn.Linear):
                    yield f"{index}.{kind}.{name}", module


@torch.no_grad()
def apply_fixed(model, bits: int, group_size: int = 128):
    """Materialize one precision for baseline runs; retain no second variant."""
    manifest = []
    for name, linear in block_linears(model):
        q, scale = quantize(linear.weight, group_size)
        linear.weight.copy_(reconstruct(q, scale, bits, linear.weight.dtype))
        manifest.append({"name": name, "shape": list(linear.weight.shape),
                         "parameters": linear.weight.numel(), "bits": bits})
    if len(manifest) != 7 * len(model.model.layers):
        raise ValueError("unexpected Qwen3 block layout")
    return manifest
