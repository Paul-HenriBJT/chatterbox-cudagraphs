"""End-to-end timing: where does a real Chatterbox generate() spend its time?

Breaks a full text->audio generation into its three phases — T3 (the
autoregressive decode loop we can CUDA-graph), S3Gen (the vocoder), and the
watermark — so we know the *ceiling* on what graphing the decode can buy
end-to-end. Graphs can only help the T3 phase, and (see caveat printed below)
only the model-forward part of it, not the per-step Python sampling.

Wraps Chatterbox's public methods with timers; does not modify its code.
"""
from __future__ import annotations

import math
import os
import statistics
import struct
import tempfile
import time
import wave

import torch

from .model import load_model

TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a test sentence of moderate length, used only to measure timing."
)


def write_dummy_wav(path: str, sr: int = 24000, secs: int = 3) -> None:
    """A short reference clip — we're timing, not judging audio quality."""
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 150 * i / sr)))
            for i in range(sr * secs)
        )
        w.writeframes(frames)


def timed(fn, acc, key, meta=None, metakey=None):
    def wrap(*a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = fn(*a, **k)
        torch.cuda.synchronize()
        acc[key] = acc.get(key, 0.0) + (time.perf_counter() - t)
        if meta is not None and metakey is not None:
            try:
                meta[metakey] = int(out.shape[-1])
            except Exception:
                pass
        return out
    return wrap


def main(runs: int = 5) -> None:
    model = load_model()

    ref = os.path.join(tempfile.gettempdir(), "ref.wav")
    write_dummy_wav(ref)
    model.prepare_conditionals(ref)

    acc, meta = {}, {}
    model.t3.inference = timed(model.t3.inference, acc, "t3", meta, "n_tokens")
    model.s3gen.inference = timed(model.s3gen.inference, acc, "s3gen")
    model.watermarker.apply_watermark = timed(model.watermarker.apply_watermark, acc, "watermark")

    model.generate(TEXT)  # warmup

    totals, t3s, s3s, wms = [], [], [], []
    for _ in range(runs):
        for k in list(acc):
            acc[k] = 0.0
        torch.cuda.synchronize()
        t = time.perf_counter()
        model.generate(TEXT)
        torch.cuda.synchronize()
        totals.append(time.perf_counter() - t)
        t3s.append(acc.get("t3", 0.0))
        s3s.append(acc.get("s3gen", 0.0))
        wms.append(acc.get("watermark", 0.0))

    total = statistics.median(totals)
    t3 = statistics.median(t3s)
    s3 = statistics.median(s3s)
    wm = statistics.median(wms)
    n = meta.get("n_tokens", 0)

    print(f"\n=== end-to-end generate() — median of {runs} runs ===")
    print(f"total     : {total*1000:8.1f} ms")
    print(f"  T3 loop : {t3*1000:8.1f} ms  ({100*t3/total:4.1f}%)   [graphable phase]")
    print(f"  S3Gen   : {s3*1000:8.1f} ms  ({100*s3/total:4.1f}%)")
    print(f"  watermk : {wm*1000:8.1f} ms  ({100*wm/total:4.1f}%)")
    print(f"  other   : {(total-t3-s3-wm)*1000:8.1f} ms  ({100*(total-t3-s3-wm)/total:4.1f}%)")
    if n:
        print(f"T3 tokens : {n}  ->  {t3/n*1000:.3f} ms/token in the real loop")
    print("\nNote: the real T3 loop includes per-step Python sampling (softmax, "
          "multinomial, CFG, penalties) that CUDA graphs do NOT remove, so the "
          "end-to-end win is bounded below the ~3x seen on the bare forward.")


if __name__ == "__main__":
    main()
