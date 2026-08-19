"""Load Chatterbox and expose ONE autoregressive T3 decode step over a static cache.

This drives Chatterbox (Resemble AI, MIT) through its public objects — it does not
vendor Chatterbox's code. It reimplements *only* the single repeated decode step,
backed by a transformers ``StaticCache`` so the step is CUDA-graph-capturable.

What the upstream T3 decode does per frame (src/chatterbox/models/t3/t3.py,
``T3.inference``):
  * classifier-free guidance -> the Llama backbone runs at batch 2 (cond+uncond)
    per stream; our ``batch_size`` multiplies that.
  * the heavy, repeated work is one ``LlamaModel`` forward over a single new-token
    embedding, then the speech head -> logits.
  * upstream keeps a growing (dynamic) cache; graph capture needs fixed addresses,
    so here we use a StaticCache. That difference is the whole point of the piece.

Timing note: launch overhead depends on the *structure* of the step (which kernels,
what shapes), not on the cache *contents* — so decoding at a fixed position over a
zero-initialised static cache is representative for the eager-vs-graph comparison.
"""
from __future__ import annotations

import torch


def load_model(device: str = "cuda", dtype: torch.dtype = torch.float16):
    """Load Chatterbox. Returns the top-level model; we use its ``.t3`` submodule.

    TODO(GPU): confirm the loader signature against the installed version.
    """
    from chatterbox.tts import ChatterboxTTS

    return ChatterboxTTS.from_pretrained(device=device)


class DecodeStep:
    def __init__(self, model, batch_size: int, device: str = "cuda",
                 dtype: torch.dtype = torch.float16, max_len: int = 1024,
                 position: int | None = None):
        t3 = model.t3
        self.backbone = t3.tfmr            # a transformers LlamaModel
        self.speech_head = t3.speech_head  # nn.Linear(dim -> speech vocab)
        self.device = device
        self.dtype = dtype

        # CFG doubles the batch (cond + uncond) per stream.
        self.eff_batch = 2 * batch_size
        self.dim = t3.dim
        self.vocab = self.speech_head.out_features
        self.max_len = max_len

        from transformers import StaticCache

        # TODO(GPU): StaticCache kwargs shifted across transformers versions
        # (max_batch_size/batch_size, max_cache_len, device, dtype). Match the pod's.
        self.cache = StaticCache(
            config=t3.cfg,
            max_batch_size=self.eff_batch,
            max_cache_len=max_len,
            device=device,
            dtype=dtype,
        )

        # Decode at a fixed mid-sequence position; only structure matters for timing.
        pos = max_len // 2 if position is None else position
        self.cache_position = torch.tensor([pos], device=device, dtype=torch.long)

        # Static, fixed-address buffers the CUDA graph will read/write.
        self.static_embed = torch.zeros(self.eff_batch, 1, self.dim,
                                        device=device, dtype=dtype)
        self.static_logits = torch.zeros(self.eff_batch, 1, self.vocab,
                                         device=device, dtype=dtype)

    def set_inputs(self, token_embed: torch.Tensor, position: int) -> None:
        """Copy the next-token embedding in and set the cache position (no realloc)."""
        self.static_embed.copy_(token_embed)
        self.cache_position.fill_(position)

    @torch.inference_mode()
    def run(self) -> None:
        out = self.backbone(
            inputs_embeds=self.static_embed,
            past_key_values=self.cache,
            cache_position=self.cache_position,
            use_cache=True,
            return_dict=True,
        )
        logits = self.speech_head(out.last_hidden_state)  # (eff_batch, 1, vocab)
        self.static_logits.copy_(logits)


if __name__ == "__main__":
    # Smoke test (needs a CUDA GPU + `pip install chatterbox-tts`).
    m = load_model()
    step = DecodeStep(m, batch_size=1)
    step.run()
    torch.cuda.synchronize()
    print("one decode step ran; logits", tuple(step.static_logits.shape))
