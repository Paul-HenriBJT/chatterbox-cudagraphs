"""Validate the CUDA-graph decode path end to end on a real GPU.

A) Correctness: graph-replay logits vs eager-step logits (max abs diff ~ 0).
B) Decode-step speedup: eager step() vs graph.replay().
C) Full sentence -> audio: wall time + real-time factor (RTF), eager vs graphed.

Writes /root/RESULT.json and prints RESULT_JSON<...>END lines for scraping.
"""
import os, time, json, tempfile, math, struct, wave
import torch

torch.set_grad_enabled(False)  # inference only; avoids 'inference tensors saved for backward' in s3gen

DEV = "cuda"
SEED = 1234
TEXT = "The quick brown fox jumps over the lazy dog, and then it does it again."
res = {"text": TEXT, "seed": SEED}

def log(*a):
    print("[val]", *a, flush=True)

def dummy_wav(path, sr=24000, secs=3):
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(
            struct.pack("<h", int(3000 * math.sin(2 * math.pi * 150 * i / sr)))
            for i in range(sr * secs)))

log("loading model...")
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device=DEV)
sr = model.sr
ref = os.path.join(tempfile.gettempdir(), "ref.wav"); dummy_wav(ref)
model.prepare_conditionals(ref)
t3 = model.t3
res["gpu"] = torch.cuda.get_device_name(0)
res["torch"] = torch.__version__
import transformers; res["transformers"] = transformers.__version__
log("gpu", res["gpu"], "torch", res["torch"], "tfmr", res["transformers"])

# capture the real t3_cond / text_tokens the pipeline builds for this text
cap = {}
_orig = t3.inference
def _spy(*a, **k):
    cap["t3_cond"] = k.get("t3_cond"); cap["text_tokens"] = k.get("text_tokens")
    return _orig(*a, **k)
t3.inference = _spy
log("warmup eager generate (also captures inputs)...")
torch.manual_seed(SEED)
wav_warm = model.generate(TEXT)
t3.inference = _orig

tfmr, head, hp = t3.tfmr, t3.speech_head, t3.hp
tt, cond = cap["text_tokens"], cap["t3_cond"]

# ---------------- A) correctness: build prefill + static cache, compare ----------------
from transformers import StaticCache
from chatterbox.models.t3.cuda_graph import _GraphedStep

initial = hp.start_speech_token * torch.ones_like(tt[:, :1])
embeds, _ = t3.prepare_input_embeds(t3_cond=cond, text_tokens=tt, speech_tokens=initial, cfg_weight=0.5)
bos = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=DEV)
bos_e = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
bos_e = torch.cat([bos_e, bos_e])
ctx = torch.cat([embeds, bos_e], dim=1)
L = ctx.size(1); dtype = ctx.dtype
cache = StaticCache(config=t3.cfg, max_batch_size=2, max_cache_len=1024, device=DEV, dtype=dtype)
pos = torch.arange(L, device=DEV)
tfmr(inputs_embeds=ctx, past_key_values=cache, cache_position=pos, use_cache=True, return_dict=True)

step_embed = torch.zeros(2, 1, ctx.size(-1), device=DEV, dtype=dtype)
step_pos = torch.tensor([L], device=DEV, dtype=torch.long)
step_logits = torch.zeros(2, 1, head.out_features, device=DEV, dtype=dtype)
def step():
    o = tfmr(inputs_embeds=step_embed, past_key_values=cache, cache_position=step_pos, use_cache=True, return_dict=True)
    step_logits.copy_(head(o.last_hidden_state))

emb = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(1)
step_embed.copy_(torch.cat([emb, emb])); step_pos.fill_(L)   # fixed position -> idempotent, comparable

log("capturing graph...")
g = _GraphedStep(step)
step(); le = step_logits.clone()          # eager forward
g.replay(); lg = step_logits.clone()      # graphed replay of the same forward
res["logit_dtype"] = str(le.dtype)
res["logit_max_abs_diff"] = (le.float() - lg.float()).abs().max().item()
res["logit_mean_abs_diff"] = (le.float() - lg.float()).abs().mean().item()
res["argmax_match"] = bool((le.argmax(-1) == lg.argmax(-1)).all().item())
log("A) logit max|diff|", res["logit_max_abs_diff"], "argmax_match", res["argmax_match"])

# ---------------- B) decode-step speedup ----------------
def bench(fn, n=200, warm=30):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0
res["step_eager_ms"] = bench(step)
res["step_graph_ms"] = bench(g.replay)
res["step_speedup"] = res["step_eager_ms"] / res["step_graph_ms"]
log("B) step eager %.3f ms  graph %.3f ms  =%.2fx" % (res["step_eager_ms"], res["step_graph_ms"], res["step_speedup"]))

# ---------------- C) full sentence -> audio, RTF ----------------
def gen_timed(use_graph, runs=3):
    if use_graph:
        o = t3.inference
        t3.inference = lambda *a, **k: o(*a, **{**k, "use_cuda_graph": True})
    torch.manual_seed(SEED); wav = model.generate(TEXT)      # warmup / graph capture
    times = []
    for _ in range(runs):
        torch.manual_seed(SEED)
        torch.cuda.synchronize(); t = time.perf_counter()
        wav = model.generate(TEXT)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    if use_graph:
        t3.inference = o
    times.sort()
    med = times[len(times)//2]
    audio_s = wav.shape[-1] / sr
    rms = float(wav.float().pow(2).mean().sqrt().item())
    return med, audio_s, rms, wav

e_t, e_audio, e_rms, wav_e = gen_timed(False)
g_t, g_audio, g_rms, wav_g = gen_timed(True)
res["eager_gen_s"] = e_t; res["eager_audio_s"] = e_audio; res["eager_rtf"] = e_t / e_audio; res["eager_rms"] = e_rms
res["graph_gen_s"] = g_t; res["graph_audio_s"] = g_audio; res["graph_rtf"] = g_t / g_audio; res["graph_rms"] = g_rms
res["e2e_speedup"] = e_t / g_t
log("C) eager %.3fs for %.2fs audio  RTF=%.3f" % (e_t, e_audio, res["eager_rtf"]))
log("C) graph %.3fs for %.2fs audio  RTF=%.3f" % (g_t, g_audio, res["graph_rtf"]))

# save audio so it can be retrieved/inspected
try:
    import torchaudio
    torchaudio.save("/root/out_eager.wav", wav_e.detach().cpu(), sr)
    torchaudio.save("/root/out_graph.wav", wav_g.detach().cpu(), sr)
except Exception as ex:
    log("wav save skipped:", ex)

with open("/root/RESULT.json", "w") as f:
    json.dump(res, f, indent=2)
print("RESULT_JSON<" + json.dumps(res) + ">END", flush=True)
log("DONE")
