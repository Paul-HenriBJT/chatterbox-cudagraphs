"""Profile one decode step: kernel count + a viewable trace.

For the article's "inside one step" figure. `torch.profiler` gives a kernel count
and a Chrome/Perfetto trace here; for a publishable timeline and a GPU-idle %,
run `nsys` as well (see README).
"""
from __future__ import annotations

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from .model import DecodeStep, load_model


def profile_step(batch_size: int = 1, device: str = "cuda",
                 trace_out: str = "results/one_step_trace.json") -> None:
    model = load_model(device=device)
    step = DecodeStep(model, batch_size=batch_size, device=device)

    for _ in range(10):
        step.run()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        step.run()
        torch.cuda.synchronize()

    prof.export_chrome_trace(trace_out)
    cuda_kernels = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
    n_kernels = sum(k.count for k in cuda_kernels)

    print(f"kernels per step ~ {n_kernels}")
    print(f"trace -> {trace_out}  (open in chrome://tracing or ui.perfetto.dev)")
    print("For a GPU-idle % and a publishable timeline, also run:")
    print("  nsys profile -o results/one_step python -m bench.profile")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--trace-out", default="results/one_step_trace.json")
    args = ap.parse_args()
    profile_step(args.batch_size, trace_out=args.trace_out)
