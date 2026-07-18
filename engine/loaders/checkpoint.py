"""
Build a `QuantizedTransformer` from a real checkpoint's raw tensors.

Two entry points:

  * `from_safetensors(path, cfg, kernels=None)`  -- HF-style .safetensors checkpoint
  * `from_gguf(path, cfg, kernels=None)`         -- llama.cpp-style .gguf checkpoint
                                                     (F16/F32 only, see gguf_io.py)

Both funnel through `build_from_tensors`, which expects a flat
`name -> np.ndarray` dict using the common HF/Llama naming convention:

    model.embed_tokens.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.norm.weight
    lm_head.weight                                  (falls back to tied embed_tokens)

GGUF tensors use llama.cpp's naming (`token_embd.weight`, `blk.{i}.attn_q.weight`,
etc.) -- `_GGUF_NAME_MAP` translates those to the same canonical names before
handing off to `build_from_tensors`, so both formats share one code path.

Every linear weight is quantized on load via `engine.quantization.get_quantizer(cfg)`
(so `cfg.quant_bits` / `cfg.quant_scheme` control the on-disk-precision ->
in-memory-precision tradeoff regardless of what precision the checkpoint
itself was stored in).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from ..config import EngineConfig
from ..kernels import FusedKernels
from ..model import QuantLinear, QuantizedTransformer, TransformerLayer
from ..quantization import get_quantizer
from . import gguf_io, safetensors_io


def _get(tensors: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name not in tensors:
        raise KeyError(f"Checkpoint is missing expected tensor '{name}'")
    return tensors[name]


def build_from_tensors(tensors: Dict[str, np.ndarray], cfg: EngineConfig,
                        kernels: Optional[FusedKernels] = None) -> QuantizedTransformer:
    """Quantize + assemble a `QuantizedTransformer` from canonically-named
    fp16/fp32 tensors (see module docstring for the naming convention)."""
    kernels = kernels or FusedKernels(cfg)
    quant = get_quantizer(cfg)

    embed = _get(tensors, "model.embed_tokens.weight").astype(np.float32)
    if embed.shape != (cfg.vocab_size, cfg.hidden_dim):
        raise ValueError(
            f"embed_tokens shape {embed.shape} doesn't match "
            f"cfg (vocab_size={cfg.vocab_size}, hidden_dim={cfg.hidden_dim}); "
            f"pass an EngineConfig that matches the checkpoint's actual shape."
        )

    layers = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        layers.append(TransformerLayer(
            attn_norm_w=_get(tensors, p + "input_layernorm.weight").astype(np.float32),
            ffn_norm_w=_get(tensors, p + "post_attention_layernorm.weight").astype(np.float32),
            wq=QuantLinear.from_weight(quant, _get(tensors, p + "self_attn.q_proj.weight")),
            wk=QuantLinear.from_weight(quant, _get(tensors, p + "self_attn.k_proj.weight")),
            wv=QuantLinear.from_weight(quant, _get(tensors, p + "self_attn.v_proj.weight")),
            wo=QuantLinear.from_weight(quant, _get(tensors, p + "self_attn.o_proj.weight")),
            w_gate=QuantLinear.from_weight(quant, _get(tensors, p + "mlp.gate_proj.weight")),
            w_up=QuantLinear.from_weight(quant, _get(tensors, p + "mlp.up_proj.weight")),
            w_down=QuantLinear.from_weight(quant, _get(tensors, p + "mlp.down_proj.weight")),
        ))

    final_norm_w = _get(tensors, "model.norm.weight").astype(np.float32)
    lm_head_w = tensors.get("lm_head.weight")
    if lm_head_w is None:
        lm_head_w = embed  # weight-tied models (common for smaller checkpoints)
    lm_head = QuantLinear.from_weight(quant, lm_head_w)

    return QuantizedTransformer(cfg, kernels, embed, layers, final_norm_w, lm_head)


def from_safetensors(path: str, cfg: EngineConfig,
                      kernels: Optional[FusedKernels] = None) -> QuantizedTransformer:
    tensors = safetensors_io.load_file(path)
    return build_from_tensors(tensors, cfg, kernels)


# llama.cpp GGUF tensor names -> this repo's canonical (HF-style) names.
# {i} is substituted per-layer.
_GGUF_NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "blk.{i}.attn_norm.weight": "model.layers.{i}.input_layernorm.weight",
    "blk.{i}.attn_q.weight": "model.layers.{i}.self_attn.q_proj.weight",
    "blk.{i}.attn_k.weight": "model.layers.{i}.self_attn.k_proj.weight",
    "blk.{i}.attn_v.weight": "model.layers.{i}.self_attn.v_proj.weight",
    "blk.{i}.attn_output.weight": "model.layers.{i}.self_attn.o_proj.weight",
    "blk.{i}.ffn_norm.weight": "model.layers.{i}.post_attention_layernorm.weight",
    "blk.{i}.ffn_gate.weight": "model.layers.{i}.mlp.gate_proj.weight",
    "blk.{i}.ffn_up.weight": "model.layers.{i}.mlp.up_proj.weight",
    "blk.{i}.ffn_down.weight": "model.layers.{i}.mlp.down_proj.weight",
}


def from_gguf(path: str, cfg: EngineConfig,
              kernels: Optional[FusedKernels] = None) -> QuantizedTransformer:
    gguf_file = gguf_io.load(path)
    tensors: Dict[str, np.ndarray] = {}
    for gguf_name in gguf_file.tensor_names():
        canonical = None
        if gguf_name in _GGUF_NAME_MAP:
            canonical = _GGUF_NAME_MAP[gguf_name]
        elif gguf_name.startswith("blk."):
            _, idx, rest = gguf_name.split(".", 2)
            template = f"blk.{{i}}.{rest}"
            if template in _GGUF_NAME_MAP:
                canonical = _GGUF_NAME_MAP[template].format(i=idx)
        if canonical is not None:
            tensors[canonical] = gguf_file.load_tensor(gguf_name)
    return build_from_tensors(tensors, cfg, kernels)


def config_from_gguf_metadata(path: str, **overrides) -> EngineConfig:
    """Best-effort: derive an `EngineConfig` from a GGUF file's own metadata
    (llama.cpp stores hidden size, layer count, head counts, etc. under
    `<arch>.*` keys), so a checkpoint's shape doesn't have to be re-typed by
    hand. Any field can still be overridden via `**overrides`."""
    gguf_file = gguf_io.load(path)
    md = gguf_file.metadata
    arch = md.get("general.architecture", "llama")

    def get(key, default):
        return md.get(f"{arch}.{key}", default)

    kwargs = dict(
        hidden_dim=get("embedding_length", 4096),
        n_layers=get("block_count", 32),
        n_heads=get("attention.head_count", 32),
        n_kv_heads=get("attention.head_count_kv", None),
        ffn_dim=get("feed_forward_length", None),
        vocab_size=len(md.get("tokenizer.ggml.tokens", [])) or 32000,
        rope_theta=get("rope.freq_base", 10000.0),
        rms_norm_eps=get("attention.layer_norm_rms_epsilon", 1e-5),
    )
    kwargs.update(overrides)
    return EngineConfig(**kwargs)
