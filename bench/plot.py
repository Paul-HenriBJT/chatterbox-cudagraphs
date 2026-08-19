"""Plot the sweep: latency (eager vs CUDA graph) + the shrinking speedup.

Two panels tell the whole story:
  left  — per-step latency; the shaded gap is the launch overhead graphs remove,
          wide at batch 1 (launch-bound) and closing by batch 64 (compute-bound).
  right — speedup (eager / graph) vs batch; it decreases because graphs remove a
          fixed per-step overhead whose share shrinks as compute grows.
"""
from __future__ import annotations

import argparse
import collections
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(csv_in: str):
    lat = collections.defaultdict(dict)
    with open(csv_in) as f:
        for r in csv.DictReader(f):
            lat[int(r["batch"])][r["mode"]] = float(r["lat_med_ms"])
    xs = sorted(lat)
    eager = [lat[x]["eager"] for x in xs]
    graph = [lat[x]["graph"] for x in xs]
    speedup = [e / g for e, g in zip(eager, graph)]
    return xs, eager, graph, speedup


def plot(csv_in: str = "results/sweep.csv", png_out: str = "results/latency.png") -> None:
    xs, eager, graph, speedup = load(csv_in)
    red, green, ink = "#c0392b", "#1e8449", "#2c3e50"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: latency, with the launch-overhead gap shaded.
    ax1.plot(xs, eager, "o-", color=red, label="eager")
    ax1.plot(xs, graph, "o-", color=green, label="CUDA graph")
    ax1.fill_between(xs, graph, eager, color=red, alpha=0.10)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([str(x) for x in xs])
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("per-step latency (ms, median)")
    ax1.set_title("The gap is launch overhead")
    ax1.legend(loc="upper left")
    ax1.annotate("launch-bound\n(GPU starved)", xy=(xs[0], (eager[0] + graph[0]) / 2),
                 fontsize=8, color=red, va="center")
    ax1.annotate("compute-bound\n(gap gone)", xy=(xs[-1], eager[-1]),
                 xytext=(xs[-1] * 0.55, eager[-1] - 6), fontsize=8, color=ink)

    # Right: the shrinking speedup.
    ax2.plot(xs, speedup, "o-", color=ink)
    ax2.axhline(1.0, ls="--", lw=1, color="gray")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([str(x) for x in xs])
    ax2.set_ylim(0.9, max(speedup) * 1.15)
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("speedup  (eager / graph)")
    ax2.set_title("The win shrinks as compute grows")
    for x, s in zip(xs, speedup):
        ax2.annotate(f"{s:.2f}×", (x, s), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=8)

    fig.suptitle("CUDA graphs vs eager — Chatterbox T3 decode step (RTX 4090)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(png_out, dpi=150)
    print(f"wrote {png_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/sweep.csv")
    ap.add_argument("--out", default="results/latency.png")
    args = ap.parse_args()
    plot(args.csv, args.out)
