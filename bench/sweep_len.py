"""Does the CUDA-graph advantage grow with sentence length?

Generates sentences of increasing length, timing the T3 decode phase (the part
graphs touch) and the full pipeline, eager vs graphed. Prints RESULT_JSON.
"""
import os, time, json, tempfile, math, struct, wave
import torch
torch.set_grad_enabled(False)

DEV = "cuda"; SEED = 1234
def log(*a): print("[sweep]", *a, flush=True)

def dummy_wav(path, sr=24000, secs=3):
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(3000*math.sin(2*math.pi*150*i/sr))) for i in range(sr*secs)))

from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device=DEV); sr = model.sr
ref = os.path.join(tempfile.gettempdir(), "ref.wav"); dummy_wav(ref); model.prepare_conditionals(ref)
t3 = model.t3

CLAUSE = "The quick brown fox jumps over the lazy dog. "
LENS = [1, 2, 4, 8]           # multiples of the clause -> increasing length
rows = []

# time the T3 decode phase by wrapping t3.inference
acc = {"t3": 0.0, "n": 0}
_orig = t3.inference
def _timed(use_graph):
    def w(*a, **k):
        if use_graph: k = {**k, "use_cuda_graph": True}
        torch.cuda.synchronize(); t = time.perf_counter()
        out = _orig(*a, **k)
        torch.cuda.synchronize(); acc["t3"] += time.perf_counter() - t
        try: acc["n"] = int(out.shape[-1])
        except Exception: pass
        return out
    return w

def run(text, use_graph, runs=2):
    t3.inference = _timed(use_graph)
    torch.manual_seed(SEED); wav = model.generate(text)      # warmup (+ capture if graph)
    e2e = []; t3t = []
    for _ in range(runs):
        acc["t3"] = 0.0
        torch.manual_seed(SEED)
        torch.cuda.synchronize(); t = time.perf_counter()
        wav = model.generate(text)
        torch.cuda.synchronize(); e2e.append(time.perf_counter() - t); t3t.append(acc["t3"])
    t3.inference = _orig
    e2e.sort(); t3t.sort()
    return e2e[len(e2e)//2], t3t[len(t3t)//2], acc["n"], wav.shape[-1]/sr

for k in LENS:
    text = (CLAUSE * k).strip()
    e_e2e, e_t3, ntok, audio = run(text, False)
    g_e2e, g_t3, ntok2, _    = run(text, True)
    row = {
        "mult": k, "n_tokens": ntok, "audio_s": round(audio, 3),
        "t3_eager_s": round(e_t3, 4), "t3_graph_s": round(g_t3, 4),
        "t3_speedup": round(e_t3 / g_t3, 3), "t3_abs_saved_s": round(e_t3 - g_t3, 4),
        "e2e_eager_s": round(e_e2e, 4), "e2e_graph_s": round(g_e2e, 4),
        "e2e_speedup": round(e_e2e / g_e2e, 3),
        "rtf_eager": round(e_e2e / audio, 3), "rtf_graph": round(g_e2e / audio, 3),
    }
    rows.append(row)
    log(k, "clause(s)", "tok", ntok, "audio %.1fs" % audio,
        "| T3 %.3f->%.3f =%.2fx (saved %.3fs)" % (e_t3, g_t3, e_t3/g_t3, e_t3-g_t3),
        "| e2e %.2fx" % (e_e2e/g_e2e))

out = {"gpu": torch.cuda.get_device_name(0), "dtype": "fp32", "rows": rows}
json.dump(out, open("/root/SWEEP.json", "w"), indent=2)
print("RESULT_JSON<" + json.dumps(out) + ">END", flush=True)
log("DONE")
