"""Must-have validation for the CUDA-graph decode path.

Item 5 — broader correctness: run the shipped `generate(use_cuda_graph=True)`
path across several texts and seeds, and confirm the waveform is identical to the
eager path (same seed).

Item 4 — fp16 vs fp32 decode step: isolated per-frame forward, graph vs eager,
logit diff + speedup, to show the dtype effect and reproduce the ~3x fp16 number.

Prints RESULT_JSON<...>END.
"""
import os, time, json, tempfile, math, struct, wave
import torch
torch.set_grad_enabled(False)

DEV = "cuda"
def log(*a): print("[mh]", *a, flush=True)

def dummy_wav(path, sr=24000, secs=3):
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(3000*math.sin(2*math.pi*150*i/sr))) for i in range(sr*secs)))

from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device=DEV); sr = model.sr
ref = os.path.join(tempfile.gettempdir(), "ref.wav"); dummy_wav(ref); model.prepare_conditionals(ref)
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
import transformers; res["transformers"] = transformers.__version__

# ---------- Item 5: broader correctness via the real generate(use_cuda_graph=) path ----------
TEXTS = [
    "Hi there.",
    "The quick brown fox jumps over the lazy dog, and then it does it again.",
    "In the beginning the universe was created. This has made a lot of people very "
    "angry and been widely regarded as a bad move, or so the story goes.",
]
SEEDS = [0, 1234]
corr = []
for ti, text in enumerate(TEXTS):
    for seed in SEEDS:
        torch.manual_seed(seed); we = model.generate(text)                     # eager
        torch.manual_seed(seed); wg = model.generate(text, use_cuda_graph=True)  # graphed (shipped flag)
        same_shape = tuple(we.shape) == tuple(wg.shape)
        maxdiff = (we - wg).abs().max().item() if same_shape else None
        row = {"text_idx": ti, "seed": seed, "eager_samples": int(we.shape[-1]),
               "graph_samples": int(wg.shape[-1]), "same_shape": same_shape,
               "wave_max_abs_diff": maxdiff, "identical": bool(same_shape and maxdiff == 0.0)}
        corr.append(row)
        log("corr t%d seed%d" % (ti, seed), "identical" if row["identical"] else row)
res["correctness"] = corr
res["all_identical"] = all(r["identical"] for r in corr)

# fallback sanity: CPU tensors / unsupported -> must not raise, returns audio
try:
    torch.manual_seed(0); _ = model.generate("Fallback check.", use_cuda_graph=True)
    res["fallback_ok"] = True
except Exception as e:
    res["fallback_ok"] = False; res["fallback_err"] = str(e)

# ---------- Item 4: isolated decode step, fp16 vs fp32 ----------
from transformers import StaticCache
from chatterbox.models.t3.cuda_graph import _GraphedStep
t3 = model.t3

def step_bench(dtype):
    tfmr, head = t3.tfmr, t3.speech_head
    if dtype == torch.float16:
        tfmr.half(); head.half()
    dim, vocab = t3.dim, head.out_features
    cache = StaticCache(config=t3.cfg, max_batch_size=2, max_cache_len=512, device=DEV, dtype=dtype)
    embed = torch.zeros(2, 1, dim, device=DEV, dtype=dtype)
    cpos = torch.tensor([256], device=DEV, dtype=torch.long)
    slog = torch.zeros(2, 1, vocab, device=DEV, dtype=dtype)
    def step():
        o = tfmr(inputs_embeds=embed, past_key_values=cache, cache_position=cpos, use_cache=True, return_dict=True)
        slog.copy_(head(o.last_hidden_state))
    g = _GraphedStep(step)
    step(); le = slog.clone(); g.replay(); lg = slog.clone()
    def bench(fn, n=200, w=30):
        for _ in range(w): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000.0
    e_ms, g_ms = bench(step), bench(g.replay)
    return {"dtype": str(dtype).replace("torch.", ""), "logit_max_diff": (le.float() - lg.float()).abs().max().item(),
            "eager_ms": round(e_ms, 3), "graph_ms": round(g_ms, 3), "speedup": round(e_ms / g_ms, 3)}

# fp32 first (no cast), then fp16 (casts the shared tfmr/head to half)
res["step_fp32"] = step_bench(torch.float32); log("fp32 step", res["step_fp32"])
res["step_fp16"] = step_bench(torch.float16); log("fp16 step", res["step_fp16"])

json.dump(res, open("/root/MUSTHAVE.json", "w"), indent=2)
print("RESULT_JSON<" + json.dumps(res) + ">END", flush=True)
log("DONE")
