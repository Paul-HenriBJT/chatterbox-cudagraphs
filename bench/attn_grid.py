"""Attention-implementation grid for Chatterbox T3's Llama backbone.

Question: does the attention implementation (eager vs SDPA/flash) actually matter
for a small autoregressive TTS, and in which regime?

Builds the exact T3 backbone (Llama_520M: 1024 hidden, 30 layers, 16 heads) with
attn_implementation="eager" vs "sdpa", and times two regimes across sequence
length and batch:
  - PREFILL: one forward over L tokens (the prompt / TTFT path) — quadratic in L,
    where flash attention should increasingly win.
  - DECODE: one new token attending a length-L KV cache (the streaming rate path)
    — a single query, where flash should barely help.

Only needs torch + transformers. Prints RESULT_JSON<...>END.
"""
import time, json, gc
import torch
from transformers import LlamaModel, LlamaConfig, StaticCache

DEV = "cuda"; DT = torch.float16
def log(*a): print("[attn]", *a, flush=True)

CFG = dict(
    vocab_size=8, max_position_embeddings=131072, hidden_size=1024,
    intermediate_size=4096, num_hidden_layers=30, num_attention_heads=16,
    num_key_value_heads=16, head_dim=64, hidden_act="silu", attention_bias=False,
    attention_dropout=0.0, mlp_bias=False, rms_norm_eps=1e-05, tie_word_embeddings=False,
    rope_theta=500000.0, use_cache=True,
    rope_scaling=dict(factor=8.0, high_freq_factor=4.0, low_freq_factor=1.0,
                      original_max_position_embeddings=8192, rope_type="llama3"),
)
HID = CFG["hidden_size"]
LENS = [128, 512, 1024, 2048]
BATCHES = [1, 8, 32]

def build(impl):
    cfg = LlamaConfig(**CFG); cfg._attn_implementation = impl
    return LlamaModel(cfg).to(DEV, DT).eval()

def bench(fn, n=50, w=10):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000.0

@torch.inference_mode()
def prefill_ms(model, B, L):
    x = torch.randn(B, L, HID, device=DEV, dtype=DT)
    return bench(lambda: model(inputs_embeds=x, use_cache=False))

@torch.inference_mode()
def decode_ms(model, B, L):
    cache = StaticCache(config=model.config, max_batch_size=B, max_cache_len=L + 1, device=DEV, dtype=DT)
    x0 = torch.randn(B, L, HID, device=DEV, dtype=DT)
    model(inputs_embeds=x0, past_key_values=cache, cache_position=torch.arange(L, device=DEV), use_cache=True)
    x1 = torch.zeros(B, 1, HID, device=DEV, dtype=DT)
    cpos = torch.tensor([L], device=DEV)
    return bench(lambda: model(inputs_embeds=x1, past_key_values=cache, cache_position=cpos, use_cache=True))

def measure(model, kind, B, L):
    torch.cuda.reset_peak_memory_stats()
    try:
        ms = prefill_ms(model, B, L) if kind == "prefill" else decode_ms(model, B, L)
        mem = torch.cuda.max_memory_allocated() / 1e9
        return {"ms": round(ms, 3), "peak_gb": round(mem, 2)}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache(); gc.collect()
        return {"ms": None, "oom": True}
    except Exception as e:
        return {"ms": None, "err": str(e)[:120]}

import transformers
res = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
       "transformers": transformers.__version__, "dtype": "fp16", "rows": []}
log(res["gpu"], "torch", res["torch"], "tfmr", res["transformers"])

models = {}
for impl in ["eager", "sdpa"]:
    log("building", impl); models[impl] = build(impl)

for kind in ["prefill", "decode"]:
    for B in BATCHES:
        for L in LENS:
            row = {"kind": kind, "batch": B, "seq_len": L}
            for impl in ["eager", "sdpa"]:
                row[impl] = measure(models[impl], kind, B, L)
            e, s = row["eager"].get("ms"), row["sdpa"].get("ms")
            row["speedup"] = round(e / s, 2) if (e and s) else None
            res["rows"].append(row)
            log(kind, "B%d L%d" % (B, L), "eager", row["eager"], "sdpa", row["sdpa"], "x", row["speedup"])
            torch.cuda.empty_cache(); gc.collect()

json.dump(res, open("/root/ATTN.json", "w"), indent=2)
print("RESULT_JSON<" + json.dumps(res) + ">END", flush=True)
log("DONE")
