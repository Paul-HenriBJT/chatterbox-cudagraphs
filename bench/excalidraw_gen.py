"""Generate Excalidraw scene files (.excalidraw) for the article diagrams.

Open the output at https://excalidraw.com (File -> Open), tweak to taste, then
export a PNG/SVG. Excalidraw renders these in its hand-drawn style; this script
just places the shapes/text/arrows so you don't start from a blank canvas.
"""
from __future__ import annotations

import json

GREEN_BG, GREEN_STK = "#b2f2bb", "#2f9e44"
RED, INK, GRAY = "#e03131", "#1e1e1e", "#868e96"

_seed = [1000]


def _base(**kw):
    _seed[0] += 1
    d = dict(
        angle=0, strokeColor=INK, backgroundColor="transparent", fillStyle="solid",
        strokeWidth=2, strokeStyle="solid", roughness=1, opacity=100, groupIds=[],
        frameId=None, roundness=None, seed=_seed[0], version=1, versionNonce=_seed[0],
        isDeleted=False, boundElements=None, updated=1, link=None, locked=False,
        id=f"el{_seed[0]}",
    )
    d.update(kw)
    return d


def rect(x, y, w, h, bg="transparent", stroke=INK, style="solid", roundness=None):
    return _base(type="rectangle", x=x, y=y, width=w, height=h, backgroundColor=bg,
                 strokeColor=stroke, strokeStyle=style, roundness=roundness)


def text(x, y, s, size=20, color=INK, family=1, align="left"):
    return _base(type="text", x=x, y=y, width=max(20, int(len(s) * size * 0.6)),
                 height=int(size * 1.25), text=s, originalText=s, fontSize=size,
                 fontFamily=family, textAlign=align, verticalAlign="top",
                 containerId=None, lineHeight=1.25, baseline=int(size * 0.8),
                 strokeColor=color)


def arrow(x, y, dx, dy, color=INK, both=False):
    return _base(type="arrow", x=x, y=y, width=abs(dx), height=abs(dy),
                 points=[[0, 0], [dx, dy]], lastCommittedPoint=None,
                 startBinding=None, endBinding=None,
                 startArrowhead="arrow" if both else None, endArrowhead="arrow",
                 strokeColor=color)


def timeline():
    els = []
    els.append(text(300, 30, "A CUDA graph removes the launch gaps", size=24))

    # eager row: green kernels separated by idle gaps (dashed red outline)
    ex, ey, kw, gap, n = 220, 170, 46, 100, 7
    for i in range(n):
        x = ex + i * gap
        els.append(rect(x, ey, kw, 50, bg=GREEN_BG, stroke=GREEN_STK))
        if i < n - 1:
            els.append(rect(x + kw, ey, gap - kw, 50, stroke=RED, style="dashed"))
    eager_end = ex + (n - 1) * gap + kw
    els.append(text(90, ey + 12, "eager", size=22))

    # graph row: kernels back-to-back
    gy = 330
    for i in range(n):
        els.append(rect(ex + i * kw, gy, kw, 50, bg=GREEN_BG, stroke=GREEN_STK))
    graph_end = ex + n * kw
    els.append(text(60, gy + 12, "CUDA graph", size=22))
    els.append(text(ex, gy + 62, "same kernels, back-to-back", size=14, color=GRAY))

    # idle annotation
    els.append(text(300, 95, "GPU idle — waiting on the CPU\nto launch the next kernel",
                    size=16, color=RED))
    els.append(arrow(420, 140, -110, 25, color=RED))

    # wall-clock saved
    els.append(arrow(graph_end, 135, eager_end - graph_end, 0, color=INK, both=True))
    els.append(text((graph_end + eager_end) / 2 - 60, 105, "wall-clock saved", size=16))
    return els


def one_step():
    els = []
    els.append(text(180, 30, "Inside one Chatterbox decode step (21 ms, batch 1)", size=22))
    total_w, x0, y0, h = 900, 180, 130, 70
    compute_w = int(total_w * 6.8 / 21.4)
    els.append(rect(x0, y0, compute_w, h, bg=GREEN_BG, stroke=GREEN_STK))
    els.append(rect(x0 + compute_w, y0, total_w - compute_w, h, bg="#ffc9c9", stroke=RED))
    els.append(text(x0 + compute_w // 2 - 60, y0 + 20, "compute ~7 ms", size=16))
    els.append(text(x0 + compute_w + (total_w - compute_w) // 2 - 150, y0 + 15,
                    "launch overhead / GPU idle  ~14 ms  (68%)", size=16, color=RED))
    els.append(text(x0, y0 + h + 20,
                    "~1,404 tiny kernels — the GPU spends ~2/3 of the step waiting",
                    size=15, color=GRAY))
    return els


def dump(name, elements):
    scene = {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
        "files": {},
    }
    with open(name, "w") as f:
        json.dump(scene, f, indent=2)
    print("wrote", name)


if __name__ == "__main__":
    dump("figures/timeline.excalidraw", timeline())
    dump("figures/one_step.excalidraw", one_step())
