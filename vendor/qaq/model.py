"""One stored quantized Qwen3 model, independent attention/FFN precision.

No CPU offload, no integer kernels. One transient FP16 variant per Linear is
cached, never serialized; it is replaced only when that block's precision changes.
The cached float computation is explicitly not low-bit resident GPU storage.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from qaq.quantization import PRECISIONS, quantize, reconstruct


class NestedLinear(nn.Module):
    def __init__(self, q, scale, bias=None):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.register_buffer("bias", bias)
        self.register_buffer("active_weight", None, persistent=False)
        self.bits = 8
        self.in_features = q.shape[1] * q.shape[2]
        self.out_features = q.shape[0]
        self.dtype = torch.float16

    @classmethod
    def from_linear(cls, linear, group_size=128):
        q, scale = quantize(linear.weight, group_size)
        module = cls(q, scale, linear.bias.detach() if linear.bias is not None else None)
        module.dtype = linear.weight.dtype
        return module

    def set_bits(self, bits):
        if bits not in PRECISIONS:
            raise ValueError(f"invalid precision: {bits}")
        if bits != self.bits:
            self.bits = bits
            self.active_weight = None

    def forward(self, inputs):
        if self.active_weight is None:
            self.active_weight = reconstruct(self.q, self.scale, self.bits, self.dtype)
        return F.linear(inputs, self.active_weight, self.bias)


def blocks(model):
    for layer in model.model.layers:
        yield layer.self_attn
        yield layer.mlp


def prepare_replacements(model, group_size=128):
    """Quantize without mutation so an independent dense reference can be tested."""
    replacements = []
    for index, block in enumerate(blocks(model)):
        count = 0
        for name, module in block.named_children():
            if isinstance(module, nn.Linear):
                packed = NestedLinear.from_linear(module, group_size)
                replacements.append((block, name, packed, index))
                count += 1
        if count != (4 if index % 2 == 0 else 3):
            raise ValueError("unexpected attention/FFN projections")
    return replacements


def install_replacements(replacements):
    for parent, name, packed, _ in replacements:
        setattr(parent, name, packed)


def set_profile(model, profile):
    block_list = list(blocks(model))
    if len(profile) != len(block_list) or any(b not in PRECISIONS for b in profile):
        raise ValueError("profile must contain one valid precision per attention and FFN block")
    # Validate before modifying anything: failed profiles are atomic.
    for block, bits in zip(block_list, profile):
        linears = [m for m in block.children() if isinstance(m, NestedLinear)]
        if not linears:
            raise ValueError("model has not been integrated")
        for linear in linears:
            linear.set_bits(bits)


def block_parameter_counts(model):
    return [sum(m.q.numel() for m in b.children() if isinstance(m, NestedLinear)) for b in blocks(model)]


def save_quantized(model, path, metadata):
    """Single checkpoint includes excluded FP16 weights and int8/scales; no variants."""
    state = model.state_dict()
    if any("active_weight" in k for k in state):
        raise AssertionError("transient cache must not be serialized")
    rotary = model.model.rotary_emb
    rotary_buffers = {name: getattr(rotary, name) for name in ("inv_freq", "original_inv_freq")}
    torch.save({"state_dict": state, "rotary_buffers": rotary_buffers,
                "config": model.config.to_dict(), "metadata": metadata}, path)


def load_quantized(path, device="cpu"):
    from transformers import AutoConfig, AutoModelForCausalLM
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config_data = dict(checkpoint["config"])
    config = AutoConfig.for_model(config_data.pop("model_type"), **config_data)
    # CPU initialization keeps nonpersistent rotary buffers valid on reload.
    # No original pretrained checkpoint or remote access is needed.
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float16,
                                           attn_implementation="sdpa")
    # Allocate shapes only; strict load below owns all persisted tensors.
    for block in blocks(model):
        for name, module in list(block.named_children()):
            if isinstance(module, nn.Linear):
                # Group size is part of the format, not inferred from model defaults.
                group_size = checkpoint["metadata"]["group_size"]
                q = torch.empty((module.out_features, module.in_features//group_size, group_size),
                                dtype=torch.int8)
                scale = torch.empty((*q.shape[:-1],1), dtype=torch.float32)
                bias = torch.empty(module.out_features, dtype=torch.float16) if module.bias is not None else None
                setattr(block, name, NestedLinear(q, scale, bias))
    model.load_state_dict(checkpoint["state_dict"], strict=True, assign=True)
    for name, tensor in checkpoint["rotary_buffers"].items():
        if name not in ("inv_freq", "original_inv_freq"):
            raise ValueError(f"unexpected rotary buffer: {name}")
        setattr(model.model.rotary_emb, name, tensor)
    model.tie_weights()
    model.requires_grad_(False)
    return model.eval().to(device), checkpoint["metadata"]
