"""Wrap a single decode step with a CUDA graph: capture once, replay many.

Generic and model-agnostic. The `step` object owns fixed-address input/output
buffers; `run()` reads/writes them in place and must not allocate or change
shapes. Copy fresh inputs into the step's buffers, then call `replay()`.
"""
from __future__ import annotations

import torch


class GraphedStep:
    def __init__(self, step, warmup: int = 3):
        self.step = step

        # Warm up on a side stream so the caching allocator and any autotuning
        # settle before capture (standard PyTorch CUDA-graph practice).
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup):
                self.step.run()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.step.run()

    def replay(self) -> None:
        self._graph.replay()
