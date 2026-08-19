# chatterbox-cudagraphs

Measuring what **CUDA graphs** save on autoregressive inference, using
[Chatterbox](https://github.com/resemble-ai/chatterbox) (Resemble AI, MIT) as the
subject — and adding an optional CUDA-graph decode path upstream.

Small autoregressive models generate one token/frame at a time, so each step is
many tiny GPU kernels. At small batch, the GPU finishes each kernel faster than
the CPU can launch the next and sits idle — you're *launch-bound*. A CUDA graph
records the step once and replays it as a single launch, removing that overhead.
This repo measures how much, honestly, on hardware you can rent for pennies.

> This wraps Chatterbox through its public API; it does not vendor its code.

## What it measures

Per batch size (`1, 8, 32, 64`), eager vs CUDA-graphed, changing nothing else:

- kernel **launches per step** (the cause),
- per-step **latency** (median, p95),
- **throughput**,
- plus a one-step **profile** (kernel count, GPU-idle %) for the trace figure.

## Reproduce it

Built to re-run cheaply. A single **RTX 4090** (~$0.4–0.7/hr on RunPod) is plenty
— Chatterbox is ~0.5B params.

```bash
git clone https://github.com/Paul-HenriBJT/chatterbox-cudagraphs
cd chatterbox-cudagraphs
bash scripts/runpod_setup.sh     # installs deps, checks the GPU
bash scripts/run_all.sh          # profile + sweep + plot -> results/
```

Outputs land in `results/`: `sweep.csv`, `latency.png`, `one_step_trace.json`.

## Method

- Eager and graphed run the **same** step; the only difference is capture/replay.
- Warm-up discarded; latency via CUDA events; multiple iters, median + p95.
- Chatterbox at ~0.5B is close to the **best case** for the technique — it's
  firmly launch-bound, so the speedup is near the ceiling for the method.
- Exact GPU / driver / CUDA / PyTorch versions are recorded in the run output.

## Layout

```
bench/graphed.py     generic CUDA-graph wrapper (capture once, replay)
bench/model.py       the only model-specific glue: one Chatterbox T3 decode step
bench/benchmark.py   the eager-vs-graph sweep
bench/profile.py     one-step profile (kernels, trace)
bench/plot.py        results chart
scripts/             RunPod setup + run-all
```

## Credits

[Chatterbox](https://github.com/resemble-ai/chatterbox) by Resemble AI (MIT).
CUDA graphs via PyTorch's `torch.cuda.CUDAGraph`.

## License

MIT.
