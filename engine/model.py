"""
A small GPT-style decoder-only transformer whose linear layers are stored
quantized (INT-N via `WeightQuantizer`, or NF4 via `NF4Quantizer` -- selected
by `EngineConfig.quant_scheme`) and executed through `engine.kernels.FusedKernels`.
Supports both single-token autoregressive decode (the path speculative
decoding and offload both optimize) and batched multi-sequence decode
(`forward_batch_step`), which amortizes the INT-N dequantization cost of each
layer's weights over every sequence in the micro-batch.

Swap `QuantizedTransformer.from_random` for `engine.loaders.checkpoint.from_safetensors`
/ `from_gguf` (see README "Extending") to run an actual trained model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import EngineConfig
from .kernels import FusedKernels
from .quantization import get_quantizer, quantize_activations_batch


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x32 = x.astype(np.float32)
    rms = np.sqrt(np.mean(x32 ** 2) + eps)
    return (x32 / rms) * weight.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


@dataclass
class QuantLinear:
    """A single quantized linear layer: y = W x."""
    qw: np.ndarray
    scales: np.ndarray
    zeros: Optional[np.ndarray]     # None for NF4 (no zero-point needed)
    out_features: int
    in_features: int

    @classmethod
    def from_random(cls, quant: Any, out_features: int, in_features: int,
                     rng: np.random.Generator) -> "QuantLinear":
        w = (rng.standard_normal((out_features, in_features)) * 0.02).astype(np.float16)
        qw, scales, zeros = quant.quantize(w)
        return cls(qw, scales, zeros, out_features, in_features)

    @classmethod
    def from_weight(cls, quant: Any, w: np.ndarray) -> "QuantLinear":
        """Quantize an already-loaded fp16/fp32 (out_features, in_features)
        weight matrix, e.g. from a real checkpoint (see `engine.loaders`)."""
        out_features, in_features = w.shape
        qw, scales, zeros = quant.quantize(w)
        return cls(qw, scales, zeros, out_features, in_features)

    def forward(self, kernels: FusedKernels, x: np.ndarray) -> np.ndarray:
        """x is a float activation vector; quantized here to INT8 per call
        (a real engine would keep a running activation scale)."""
        x32 = x.astype(np.float32)
        amax = np.max(np.abs(x32)) + 1e-8
        x_scale = amax / 127.0
        x_q = np.clip(np.round(x32 / x_scale), -127, 127).astype(np.int8)
        y = kernels.fused_int4_gemv(self.qw, self.scales, self.zeros, x_q, x_scale)
        return y.astype(np.float32)

    def forward_batch(self, kernels: FusedKernels, X: np.ndarray) -> np.ndarray:
        """X shape (batch, in_features) -> returns (batch, out_features).
        Dequantizes this layer's weight matrix exactly once regardless of
        batch size (see `FusedKernels.fused_int4_gemm_batch`)."""
        Xq, x_scales = quantize_activations_batch(X)
        y = kernels.fused_int4_gemm_batch(self.qw, self.scales, self.zeros, Xq, x_scales)
        return y.astype(np.float32)


@dataclass
class TransformerLayer:
    attn_norm_w: np.ndarray
    ffn_norm_w: np.ndarray
    wq: QuantLinear
    wk: QuantLinear
    wv: QuantLinear
    wo: QuantLinear
    w_gate: QuantLinear
    w_up: QuantLinear
    w_down: QuantLinear

    @classmethod
    def from_random(cls, cfg: EngineConfig, quant: Any, rng: np.random.Generator) -> "TransformerLayer":
        h, ffn = cfg.hidden_dim, cfg.ffn_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        return cls(
            attn_norm_w=np.ones(h, dtype=np.float32),
            ffn_norm_w=np.ones(h, dtype=np.float32),
            wq=QuantLinear.from_random(quant, h, h, rng),
            wk=QuantLinear.from_random(quant, kv_dim, h, rng),
            wv=QuantLinear.from_random(quant, kv_dim, h, rng),
            wo=QuantLinear.from_random(quant, h, h, rng),
            w_gate=QuantLinear.from_random(quant, ffn, h, rng),
            w_up=QuantLinear.from_random(quant, ffn, h, rng),
            w_down=QuantLinear.from_random(quant, h, ffn, rng),
        )


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


class KVCache:
    """Per-layer key/value cache, grown incrementally during decode."""

    def __init__(self, n_layers: int):
        self.k: List[List[np.ndarray]] = [[] for _ in range(n_layers)]
        self.v: List[List[np.ndarray]] = [[] for _ in range(n_layers)]

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray):
        self.k[layer_idx].append(k)
        self.v[layer_idx].append(v)

    def get(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return np.stack(self.k[layer_idx]), np.stack(self.v[layer_idx])

    def length(self, layer_idx: int) -> int:
        return len(self.k[layer_idx])

    def truncate(self, layer_idx: int, length: int):
        self.k[layer_idx] = self.k[layer_idx][:length]
        self.v[layer_idx] = self.v[layer_idx][:length]


class QuantizedTransformer:
    """A compact decoder-only transformer, one token in -> logits out."""

    def __init__(self, cfg: EngineConfig, kernels: FusedKernels,
                 embed: np.ndarray, layers: List[TransformerLayer],
                 final_norm_w: np.ndarray, lm_head: QuantLinear):
        self.cfg = cfg
        self.kernels = kernels
        self.embed = embed              # (vocab_size, hidden_dim), fp32
        self.layers = layers
        self.final_norm_w = final_norm_w
        self.lm_head = lm_head

    @classmethod
    def from_random(cls, cfg: EngineConfig, kernels: Optional[FusedKernels] = None) -> "QuantizedTransformer":
        """Build a randomly-initialized model of the given shape. Useful for
        smoke-testing the pipeline without a real checkpoint."""
        rng = np.random.default_rng(cfg.seed)
        kernels = kernels or FusedKernels(cfg)
        quant = get_quantizer(cfg)

        embed = (rng.standard_normal((cfg.vocab_size, cfg.hidden_dim)) * 0.02).astype(np.float32)
        layers = [TransformerLayer.from_random(cfg, quant, rng) for _ in range(cfg.n_layers)]
        final_norm_w = np.ones(cfg.hidden_dim, dtype=np.float32)
        lm_head = QuantLinear.from_random(quant, cfg.vocab_size, cfg.hidden_dim, rng)
        return cls(cfg, kernels, embed, layers, final_norm_w, lm_head)

    # ------------------------------------------------------------------ #
    def forward_token(self, token_id: int, pos: int, cache: KVCache,
                       layer_provider=None) -> np.ndarray:
        """Run one decode step. `layer_provider(i)` optionally overrides how
        layer `i`'s weights are obtained (e.g. via an OffloadManager)."""
        cfg = self.cfg
        x = self.embed[token_id].copy()

        for i, layer in enumerate(self.layers):
            layer = layer_provider(i) if layer_provider is not None else layer

            normed = _rmsnorm(x, layer.attn_norm_w, cfg.rms_norm_eps)
            q = self.kernels._apply_rope(layer.wq.forward(self.kernels, normed), pos)
            k = self.kernels._apply_rope(layer.wk.forward(self.kernels, normed), pos)
            v = layer.wv.forward(self.kernels, normed)
            cache.append(i, k, v)

            attn_out = self._attention(q, cache, i)
            x = x + layer.wo.forward(self.kernels, attn_out)

            normed2 = _rmsnorm(x, layer.ffn_norm_w, cfg.rms_norm_eps)
            gate = _silu(layer.w_gate.forward(self.kernels, normed2))
            up = layer.w_up.forward(self.kernels, normed2)
            x = x + layer.w_down.forward(self.kernels, gate * up)

        x = _rmsnorm(x, self.final_norm_w, cfg.rms_norm_eps)
        logits = self.lm_head.forward(self.kernels, x)
        return logits

    # ------------------------------------------------------------------ #
    # batched multi-sequence decode: batches every linear-layer GEMM across
    # sequences (the expensive, dequant-dominated part) while keeping
    # attention itself per-sequence, since each sequence has its own KV
    # cache and possibly different length. This is the standard shape of
    # real batched-decode engines: attention is inherently per-sequence,
    # everything else vectorizes trivially over the batch dimension.
    # ------------------------------------------------------------------ #
    def forward_batch_step(self, token_ids: List[int], pos: int, caches: List["KVCache"],
                            layer_provider=None) -> np.ndarray:
        """One decode step for `batch = len(token_ids)` independent sequences,
        all currently at the same position `pos`. Returns logits, shape
        (batch, vocab_size)."""
        cfg = self.cfg
        batch = len(token_ids)
        assert len(caches) == batch
        X = np.stack([self.embed[t] for t in token_ids]).astype(np.float32)   # (batch, hidden)

        for i, layer in enumerate(self.layers):
            layer = layer_provider(i) if layer_provider is not None else layer

            normed = self._rmsnorm_batch(X, layer.attn_norm_w, cfg.rms_norm_eps)
            Q = self.kernels._apply_rope_batch(layer.wq.forward_batch(self.kernels, normed), pos)
            K = self.kernels._apply_rope_batch(layer.wk.forward_batch(self.kernels, normed), pos)
            V = layer.wv.forward_batch(self.kernels, normed)

            attn_outs = np.empty((batch, cfg.hidden_dim), dtype=np.float32)
            for b in range(batch):
                caches[b].append(i, K[b], V[b])
                attn_outs[b] = self._attention(Q[b], caches[b], i)

            X = X + layer.wo.forward_batch(self.kernels, attn_outs)

            normed2 = self._rmsnorm_batch(X, layer.ffn_norm_w, cfg.rms_norm_eps)
            gate = _silu(layer.w_gate.forward_batch(self.kernels, normed2))
            up = layer.w_up.forward_batch(self.kernels, normed2)
            X = X + layer.w_down.forward_batch(self.kernels, gate * up)

        X = self._rmsnorm_batch(X, self.final_norm_w, cfg.rms_norm_eps)
        logits = self.lm_head.forward_batch(self.kernels, X)   # (batch, vocab)
        return logits

    @staticmethod
    def _rmsnorm_batch(X: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
        rms = np.sqrt(np.mean(X ** 2, axis=1, keepdims=True) + eps)
        return (X / rms) * weight.astype(np.float32)[None, :]

    def _attention(self, q: np.ndarray, cache: "KVCache", layer_idx: int) -> np.ndarray:
        """Vectorized grouped-query attention for a single sequence/timestep:
        every head is scored and mixed in one batch of `einsum` calls instead
        of a Python loop over heads."""
        cfg = self.cfg
        head_dim = cfg.head_dim
        n_heads = cfg.n_heads
        n_kv_heads = cfg.n_kv_heads
        group = n_heads // n_kv_heads

        ks, vs = cache.get(layer_idx)                              # (seq, kv_dim)
        seq_len = ks.shape[0]
        scale = 1.0 / np.sqrt(head_dim)

        q_h = q.reshape(n_heads, head_dim)                          # (n_heads, head_dim)
        ks_h = ks.reshape(seq_len, n_kv_heads, head_dim)
        vs_h = vs.reshape(seq_len, n_kv_heads, head_dim)
        if group > 1:
            ks_h = np.repeat(ks_h, group, axis=1)                   # (seq, n_heads, head_dim)
            vs_h = np.repeat(vs_h, group, axis=1)

        scores = np.einsum("hd,shd->sh", q_h, ks_h) * scale         # (seq, n_heads)
        scores = scores - scores.max(axis=0, keepdims=True)
        weights = np.exp(scores)
        weights /= weights.sum(axis=0, keepdims=True)
        out = np.einsum("sh,shd->hd", weights, vs_h)                # (n_heads, head_dim)
        return out.reshape(-1)

    def sample(self, logits: np.ndarray, temperature: float = 1.0,
               rng: Optional[np.random.Generator] = None) -> Tuple[int, np.ndarray]:
        """Returns (sampled_token_id, full probability distribution)."""
        rng = rng or np.random.default_rng()
        if temperature <= 1e-6:
            probs = np.zeros_like(logits)
            probs[np.argmax(logits)] = 1.0
        else:
            probs = _softmax(logits / temperature)
        token = int(rng.choice(len(probs), p=probs))
        return token, probs
