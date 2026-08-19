"""Generate the article's explanatory figures.

- timeline.png : schematic (illustrative) — eager launches leave GPU-idle gaps;
                 a CUDA graph replays the same kernels back-to-back.
- one_step.png : data-driven — one real decode step (21 ms) split into GPU
                 compute (~7 ms) vs launch overhead / idle (~14 ms, ~68%).
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

GREEN, RED, GRAY, INK = "#1e8449", "#c0392b", "#95a5a6", "#2c3e50"


def timeline(out: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.0))
    kw, gap, n = 0.5, 1.15, 7

    x = 0.0
    for i in range(n):
        ax.add_patch(Rectangle((x, 1.45), kw, 0.5, color=GREEN))
        if i < n - 1:
            ax.add_patch(Rectangle((x + kw, 1.45), gap - kw, 0.5,
                                    fill=False, ec=RED, ls=(0, (2, 2)), lw=1))
        x += gap
    eager_end = x - (gap - kw)

    x2 = 0.0
    for _ in range(n):
        ax.add_patch(Rectangle((x2, 0.45), kw, 0.5, color=GREEN))
        x2 += kw
    graph_end = x2

    ax.text(-0.25, 1.70, "eager", ha="right", va="center", fontsize=11, weight="bold")
    ax.text(-0.25, 0.70, "CUDA graph", ha="right", va="center", fontsize=11, weight="bold")
    ax.annotate("GPU idle — waiting on the CPU\nto launch the next kernel",
                xy=(kw + (gap - kw) / 2, 1.95), xytext=(1.6, 2.55),
                fontsize=9, color=RED, arrowprops=dict(arrowstyle="->", color=RED))
    ax.annotate("", xy=(graph_end, 2.15), xytext=(eager_end, 2.15),
                arrowprops=dict(arrowstyle="<->", color=INK))
    ax.text((graph_end + eager_end) / 2, 2.25, "wall-clock saved",
            ha="center", fontsize=9, color=INK)
    ax.text(graph_end / 2, 0.18, "same kernels, back-to-back",
            ha="center", fontsize=8, color=GRAY)

    ax.set_xlim(-2.2, eager_end + 0.4)
    ax.set_ylim(0, 2.8)
    ax.axis("off")
    ax.set_title("A CUDA graph removes the launch gaps — same compute, less wall-clock",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def one_step(out: str, total: float = 21.4, compute: float = 6.8) -> None:
    overhead = total - compute
    fig, ax = plt.subplots(figsize=(9, 2.1))
    ax.barh(0, compute, color=GREEN,
            label=f"GPU compute   {compute:.0f} ms  ({100 * compute / total:.0f}%)")
    ax.barh(0, overhead, left=compute, color=RED, alpha=0.9,
            label=f"launch overhead / GPU idle   {overhead:.0f} ms  ({100 * overhead / total:.0f}%)")
    ax.text(compute / 2, 0, f"{compute:.0f} ms", ha="center", va="center",
            color="white", fontsize=10, weight="bold")
    ax.text(compute + overhead / 2, 0, f"{overhead:.0f} ms", ha="center", va="center",
            color="white", fontsize=10, weight="bold")
    ax.set_xlim(0, total * 1.02)
    ax.set_ylim(-0.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("one decode step (ms) — batch 1, RTX 4090")
    ax.set_title("Inside one Chatterbox step: ~1,404 tiny kernels, ~68% of the time GPU-idle",
                 fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=1, frameon=False, fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", default="results/timeline.png")
    ap.add_argument("--onestep", default="results/one_step.png")
    args = ap.parse_args()
    timeline(args.timeline)
    one_step(args.onestep)
