# -*- coding: utf-8 -*-
"""Complexity profiling (§3.16): parameter count, FLOPs, inference time,
peak memory. Hook-based FLOP counting for the layer types used here
(Conv1d, Linear, GRU) — no external dependency."""

import time
from typing import Dict

import torch
from torch import nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def estimate_flops(model: nn.Module, example_inputs: dict) -> int:
    """Multiply-accumulate count (x2 = FLOPs) for one forward pass."""
    macs = [0]
    hooks = []

    def conv_hook(module, inputs, output):
        out_elems = output.numel()
        k = module.kernel_size[0]
        macs[0] += out_elems * module.in_channels * k // module.groups

    def linear_hook(module, inputs, output):
        macs[0] += output.numel() * module.in_features

    def gru_hook(module, inputs, output):
        x = inputs[0]
        steps = x.shape[1] if module.batch_first else x.shape[0]
        batch = x.shape[0] if module.batch_first else x.shape[1]
        dirs = 2 if module.bidirectional else 1
        h = module.hidden_size
        layer_macs = 0
        in_dim = module.input_size
        for layer in range(module.num_layers):
            layer_macs += 3 * h * (in_dim + h)  # 3 gates, input+recurrent
            in_dim = h * dirs
        macs[0] += layer_macs * steps * batch * dirs

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.GRU):
            hooks.append(m.register_forward_hook(gru_hook))

    model.eval()
    with torch.no_grad():
        model(**example_inputs)
    for h in hooks:
        h.remove()
    return 2 * macs[0]


def measure_inference(model: nn.Module, example_inputs: dict,
                      n_warmup: int = 2, n_runs: int = 10) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**example_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            model(**example_inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / n_runs
    peak_mem = (torch.cuda.max_memory_allocated()
                if device.type == "cuda" else 0)
    return {"inference_sec": elapsed, "peak_memory_bytes": float(peak_mem)}


def profile_model(model: nn.Module, batch_size: int = 1, seq_len: int = 85,
                  n_channels: int = 2, epoch_samples: int = 3000) -> dict:
    device = next(model.parameters()).device
    example = {
        "x": torch.zeros(batch_size, seq_len, n_channels, epoch_samples,
                         device=device),
        "tso": torch.linspace(0, 1, seq_len, device=device
                              ).unsqueeze(0).expand(batch_size, -1),
        "channel_mask": torch.ones(batch_size, seq_len, n_channels,
                                   dtype=torch.bool, device=device),
    }
    out = {"parameters": count_parameters(model),
           "flops_forward": estimate_flops(model, example),
           "profile_input": {"batch_size": batch_size, "seq_len": seq_len,
                             "n_channels": n_channels,
                             "epoch_samples": epoch_samples}}
    out.update(measure_inference(model, example))
    return out
