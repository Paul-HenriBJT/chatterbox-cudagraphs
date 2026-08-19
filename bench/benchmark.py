"""Sweep batch sizes; time one decode step eager vs CUDA-graphed; count launches.

Eager and graphed run the *same* step; the only difference is capture/replay.
Latency via CUDA events; kernel-launch count via the profiler.
"""
from __future__ import annotations

import argparse
import csv
import gc
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

from .graphed import GraphedStep
from .model import DecodeStep, load_model


def time_step(run, iters: int = 200, warmup: int = 20) -> tuple[float, float]:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    lat_ms = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        lat_ms.append(start.elapsed_time(end))
    lat_ms.sort()
    return statistics.median(lat_ms), lat_ms[int(0.95 * len(lat_ms)) - 1]


def count_launches(run) -> int:
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    # One kernel dispatch per launch call. Names vary by CUDA version.
    launch_names = {"cudaLaunchKernel", "cudaLaunchKernelExC"}
    return sum(1 for e in prof.events() if e.name in launch_names)


def run_sweep(batch_sizes, out_csv: str, device: str = "cuda", max_len: int = 512) -> None:
    model = load_model(device=device)
    rows = []
    for bs in batch_sizes:
        step = graphed = None
        try:
            step = DecodeStep(model, batch_size=bs, device=device, max_len=max_len)

            med, p95 = time_step(step.run)
            launches = count_launches(step.run)
            rows.append(dict(batch=bs, mode="eager", launches=launches,
                             lat_med_ms=round(med, 4), lat_p95_ms=round(p95, 4)))

            graphed = GraphedStep(step)
            gmed, gp95 = time_step(graphed.replay)
            rows.append(dict(batch=bs, mode="graph", launches=1,
                             lat_med_ms=round(gmed, 4), lat_p95_ms=round(gp95, 4)))

            print(f"bs={bs:>3}: eager {med:7.3f} ms ({launches} launches)  ->  "
                  f"graph {gmed:7.3f} ms  ({med / gmed:.2f}x)")
        except torch.cuda.OutOfMemoryError:
            print(f"bs={bs:>3}: OOM (skipped)")
        finally:
            # Free the cache + captured graph before the next size (they accumulate).
            del graphed, step
            gc.collect()
            torch.cuda.empty_cache()

    if not rows:
        print("no rows collected")
        return
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32, 64])
    ap.add_argument("--out", default="results/sweep.csv")
    args = ap.parse_args()
    run_sweep(args.batch_sizes, args.out)
