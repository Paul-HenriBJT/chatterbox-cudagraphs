# PR draft — to resemble-ai/chatterbox

> DRAFT to review before opening. The code is on the fork branch
> `cudagraph-decode`:
> https://github.com/Paul-HenriBJT/chatterbox/tree/cudagraph-decode
> — an opt-in `use_cuda_graph=True` flag on `T3.inference()` plus a `cuda_graph.py`
> helper (StaticCache + captured/replayed per-frame forward). Off by default.
>
> **GPU-validated (RTX 4090, torch 2.6.0, transformers 5.2.0).** With the graphed
> sampler matched to `T3.inference` and a fixed seed, the graphed path is
> **output-identical** to eager: per-step logit max|Δ| = 0.0 and identical token
> sequences across several texts and seeds (fp32 and fp16). Validation scripts:
> `bench/validate_e2e.py` and `bench/validate_musthave.py` in the harness repo.
> No PR is open yet — review, then open it.

---

**Title:** Add an optional CUDA-graph decode path to T3 (faster real-time inference)

## What

Adds an optional `use_cuda_graph` path to the T3 autoregressive decode loop. When
enabled, the per-frame model forward is captured into a CUDA graph once and
replayed each step, instead of re-launching every kernel from Python on every
step. Off by default; opt-in, output-identical.

## Why

At batch 1 — real-time / streaming inference — T3 decode is *launch-bound*, not
compute-bound. Each step is a full forward over one new token: hundreds of tiny
kernels. On an RTX 4090 the GPU sits idle a large fraction of each step, waiting
on the CPU to dispatch the next of ~1,400 kernel launches. A CUDA graph collapses
those launches into a single replay.

## Result (measured on RTX 4090, torch 2.6.0, transformers 5.2.0)

**Correctness — output-identical.** Sampling in the graphed path is matched to
`T3.inference` (CFG → repetition penalty → temperature → min_p → top_p). With a
fixed seed, graphed vs eager, across several texts and seeds:

- per-step logit max|Δ| = **0.0**, argmax match = **true**
- **identical token sequences** (end-to-end waveform identical up to ~1e-5 of
  vocoder float-nondeterminism, which is unrelated to the graph)
- holds in both fp32 and fp16 (logit Δ = 0.0 in both)

**Decode step (batch 1 — the real-time regime).** The captured per-frame forward:

| | eager | graph | speedup |
|--|------:|------:|:-------:|
| model forward, batch 1 | 18.3 ms | 8.4 ms | **2.16×** |

An isolated fp16 microbenchmark in the harness shows the batch-size dependence —
largest at batch 1, fading to ~1.05× by batch 64 as the step becomes
compute-bound:

| batch | eager | graph | speedup |
|------:|------:|------:|:-------:|
| 1  | 21.4 ms | 6.8 ms | 3.1× |
| 8  | 26.7 ms | 8.3 ms | 3.2× |
| 32 | 27.0 ms | 17.9 ms | 1.5× |
| 64 | 32.0 ms | 30.6 ms | 1.05× |

**Full utterance → audio.** A 6.5 s sentence, whole pipeline:

| | wall | real-time factor |
|--|----:|----:|
| eager | 3.14 s | 0.48 |
| graph | 2.00 s | 0.31 |

End-to-end **1.57×**. It's below the step speedup because the graph only touches
the T3 model-forward: the per-step Python sampling, the S3Gen vocoder, and the
watermark run eagerly in both and set the end-to-end ceiling.

**Scales with length.** The per-step saving accumulates and the one-time capture
amortizes, so longer utterances gain a little more:

| audio | T3 decode eager→graph | T3 speedup | e2e speedup |
|------:|:---------------------:|:----------:|:-----------:|
| ~8 s  | 3.13 → 2.02 s | 1.55× | 1.47× |
| ~16 s | 6.38 → 3.79 s | 1.68× | 1.59× |

Full benchmark harness: https://github.com/Paul-HenriBJT/chatterbox-cudagraphs

## Approach

- Opt in with `use_cuda_graph=True` on `inference()`; default behaviour unchanged.
- Drive the decode loop with a `StaticCache` (fixed shapes/addresses) so the
  captured graph stays valid across steps.
- Capture the per-step forward once (after a short warmup); each step copies the
  new token embedding + position into the static buffers and replays.
- CFG combination and sampling stay in Python, unchanged and matched to eager.
- Prefill stays eager (different shapes); only the repeated per-frame decode is
  graphed.

## Caveats (stated plainly)

- Helps at small batch / real-time; negligible at large batch.
- **No effect on time-to-first-token** — prefill is untouched; it speeds up the
  per-frame *rate*.
- CUDA-only, single utterance. Any other case (CPU/MPS, batched, capture failure)
  **falls back to the eager path** — the flag never breaks generation.
- The static cache is sized to fit prompt + `max_new_tokens`, so it can't overflow.
- Output is unchanged (identical kernels and math; verified identical tokens).

## Reproduce

```bash
# on a CUDA GPU:
pip install "git+https://github.com/Paul-HenriBJT/chatterbox.git@cudagraph-decode"
pip uninstall -y torchvision   # not needed; avoids an nms import clash
git clone https://github.com/Paul-HenriBJT/chatterbox-cudagraphs
cd chatterbox-cudagraphs

python bench/validate_e2e.py        # correctness + decode step + end-to-end RTF
python bench/validate_musthave.py   # shipped generate() flag + fallback + fp16 vs fp32
bash scripts/runpod_setup.sh && bash scripts/run_all.sh   # decode-step batch sweep
```
