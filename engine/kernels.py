"""
Fused compute kernels.

Two operations dominate decode-time cost in an offloaded, quantized
transformer:

  1. INT4-weight x INT8-activation GEMV (one token at a time during decode)
  2. RMSNorm immediately followed by RoPE rotation on the normed output

Fusing each pair avoids materializing an intermediate full-precision tensor
and avoids a second kernel launch / memory round-trip. This module exposes a
single NumPy implementation that is always correct, and transparently swaps
in the compiled CUDA kernels (`cuda/fused_int4_gemv.cu`,
`cuda/fused_rms_norm_rope.cu`) through CuPy's RawKernel API when a GPU and a
matching `.cubin`/`.ptx` build are available. This means the engine runs
identically (just slower) on a CPU-only dev machine.
"""
from __future__ import annotations

import os
import numpy as np

from .config import EngineConfig
from .quantization import get_quantizer

_CUDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")


def _try_load_cupy():
    try:
        import cupy as cp  # noqa: F401
        return cp
    except Exception:
        return None


class FusedKernels:
    """Dispatches to CUDA RawKernels when available, else pure NumPy."""

    def __init__(self, config: EngineConfig, force_numpy: bool = False):
        self.cfg = config
        self.quant = get_quantizer(config)
        self._cp = None if force_numpy else _try_load_cupy()
        self._gemv_kernel = None
        self._norm_rope_kernel = None
        if self._cp is not None:
            self._try_compile_cuda()

    @property
    def using_cuda(self) -> bool:
        return self._gemv_kernel is not None

    # ------------------------------------------------------------------ #
    def _try_compile_cuda(self):
        """Best-effort JIT compile of the .cu sources via CuPy RawModule."""
        try:
            gemv_src = open(os.path.join(_CUDA_DIR, "fused_int4_gemv.cu")).read()
            norm_src = open(os.path.join(_CUDA_DIR, "fused_rms_norm_rope.cu")).read()
            self._gemv_kernel = self._cp.RawKernel(gemv_src, "fused_int4_gemv")
            self._norm_rope_kernel = self._cp.RawKernel(norm_src, "fused_rms_norm_rope")
        except Exception:
            # No nvcc / no GPU / compile error -> silently fall back to NumPy.
            self._gemv_kernel = None
            self._norm_rope_kernel = None

    # ------------------------------------------------------------------ #
    # fused INT4 dequant + GEMV
    # ------------------------------------------------------------------ #
    def fused_int4_gemv(self, qw_packed: np.ndarray, scales: np.ndarray,
                         zeros: np.ndarray, x: np.ndarray, x_scale: float) -> np.ndarray:
        """y = dequant(qw) @ (x * x_scale)

        qw_packed : uint8, shape (out_features, in_features // 2)
        scales/zeros : float32, shape (out_features, n_groups)
        x         : int8, shape (in_features,)   -- quantized activations
        x_scale   : float, dequantization scale for x
        returns   : float16, shape (out_features,)
        """
        if self.using_cuda:
            return self._fused_int4_gemv_cuda(qw_packed, scales, zeros, x, x_scale)
        return self._fused_int4_gemv_numpy(qw_packed, scales, zeros, x, x_scale)

    def _fused_int4_gemv_numpy(self, qw_packed, scales, zeros, x, x_scale) -> np.ndarray:
        w = self.quant.dequantize(qw_packed, scales, zeros)          # (out, in) float32
        x_f = x.astype(np.float32) * np.float32(x_scale)             # (in,)
        y = w @ x_f                                                  # (out,)
        return y.astype(np.float16)

    def _fused_int4_gemv_cuda(self, qw_packed, scales, zeros, x, x_scale) -> np.ndarray:
        cp = self._cp
        out_features, packed_in = qw_packed.shape
        in_features = packed_in * self.quant.values_per_byte
        n_groups = scales.shape[1]
        group_size = self.cfg.group_size

        qw_d = cp.asarray(qw_packed)
        sc_d = cp.asarray(scales, dtype=cp.float32)
        zr_d = cp.asarray(zeros, dtype=cp.float32)
        x_d = cp.asarray(x, dtype=cp.int8)
        y_d = cp.zeros((out_features,), dtype=cp.float32)

        threads = 128
        blocks = out_features
        self._gemv_kernel(
            (blocks,), (threads,),
            (qw_d, sc_d, zr_d, x_d, cp.float32(x_scale), y_d,
             cp.int32(out_features), cp.int32(in_features),
             cp.int32(group_size), cp.int32(n_groups)),
        )
        return cp.asnumpy(y_d).astype(np.float16)

    # ------------------------------------------------------------------ #
    # fused INT4 dequant + batched GEMM (prefill / multi-sequence decode)
    # ------------------------------------------------------------------ #
    def fused_int4_gemm_batch(self, qw_packed: np.ndarray, scales: np.ndarray,
                               zeros: np.ndarray, X: np.ndarray, x_scales: np.ndarray) -> np.ndarray:
        """Y = dequant(qw) @ (X * x_scales[:, None]).T, batched over rows of X.

        qw_packed/scales/zeros : as in `fused_int4_gemv`
        X         : int8, shape (batch, in_features)
        x_scales  : float32, shape (batch,)  -- per-row dequant scale
        returns   : float16, shape (batch, out_features)

        The key efficiency win over calling `fused_int4_gemv` in a Python
        loop: `dequant(qw)` -- the expensive O(out * in) unpack+affine step --
        happens exactly once per layer regardless of batch size, and the
        actual matmul is a single BLAS call instead of `batch` separate ones.
        """
        w = self.quant.dequantize(qw_packed, scales, zeros)              # (out, in) float32, once
        Xf = X.astype(np.float32) * x_scales.astype(np.float32)[:, None]  # (batch, in)
        Y = Xf @ w.T                                                      # (batch, out)
        return Y.astype(np.float16)

    # ------------------------------------------------------------------ #
    # fused RMSNorm + RoPE
    # ------------------------------------------------------------------ #
    def fused_rmsnorm_rope(self, x: np.ndarray, weight: np.ndarray, pos: int,
                            eps: float | None = None) -> np.ndarray:
        """RMSNorm(x) * weight, then apply RoPE rotation at position `pos`.

        x, weight : float32/float16, shape (hidden_dim,)
        returns   : float32, shape (hidden_dim,)
        """
        if self.using_cuda:
            return self._fused_rmsnorm_rope_cuda(x, weight, pos, eps)
        return self._fused_rmsnorm_rope_numpy(x, weight, pos, eps)

    def fused_rmsnorm_rope_batch(self, X: np.ndarray, weight: np.ndarray, pos: int,
                                  eps: float | None = None) -> np.ndarray:
        """Batched version: X shape (batch, hidden_dim), same `pos` for every
        row (used during batched prefill/decode where all sequences in the
        micro-batch are at the same timestep)."""
        eps = self.cfg.rms_norm_eps if eps is None else eps
        X32 = X.astype(np.float32)
        rms = np.sqrt(np.mean(X32 ** 2, axis=1, keepdims=True) + eps)     # (batch, 1)
        normed = (X32 / rms) * weight.astype(np.float32)[None, :]
        return self._apply_rope_batch(normed, pos)

    def _fused_rmsnorm_rope_numpy(self, x, weight, pos, eps) -> np.ndarray:
        eps = self.cfg.rms_norm_eps if eps is None else eps
        x32 = x.astype(np.float32)
        rms = np.sqrt(np.mean(x32 ** 2) + eps)
        normed = (x32 / rms) * weight.astype(np.float32)
        return self._apply_rope(normed, pos)

    def _fused_rmsnorm_rope_cuda(self, x, weight, pos, eps) -> np.ndarray:
        cp = self._cp
        eps = self.cfg.rms_norm_eps if eps is None else eps
        hidden_dim = x.shape[0]
        head_dim = self.cfg.head_dim
        x_d = cp.asarray(x, dtype=cp.float32)
        w_d = cp.asarray(weight, dtype=cp.float32)
        out_d = cp.zeros_like(x_d)
        self._norm_rope_kernel(
            (hidden_dim // head_dim,), (head_dim,),
            (x_d, w_d, out_d, cp.int32(hidden_dim), cp.int32(head_dim),
             cp.int32(pos), cp.float32(eps), cp.float32(self.cfg.rope_theta)),
        )
        return cp.asnumpy(out_d)

    def _apply_rope(self, x: np.ndarray, pos: int) -> np.ndarray:
        """Rotary position embedding applied per attention head, vectorized
        across all heads at once (reshape to (n_heads, head_dim) instead of
        a Python loop)."""
        head_dim = self.cfg.head_dim
        n_heads = x.shape[0] // head_dim
        half = head_dim // 2
        cos, sin = self._rope_cos_sin(pos)                    # (half,)

        xr = x.reshape(n_heads, head_dim)
        x1, x2 = xr[:, :half], xr[:, half:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return np.concatenate([out1, out2], axis=1).reshape(-1)

    def _apply_rope_batch(self, X: np.ndarray, pos: int) -> np.ndarray:
        """Batched RoPE: X shape (batch, hidden_dim), same position for every row."""
        head_dim = self.cfg.head_dim
        batch, hidden_dim = X.shape
        n_heads = hidden_dim // head_dim
        half = head_dim // 2
        cos, sin = self._rope_cos_sin(pos)                    # (half,)

        Xr = X.reshape(batch, n_heads, head_dim)
        x1, x2 = Xr[:, :, :half], Xr[:, :, half:]
        out1 = x1 * cos - x2 * sin
        out2 = x2 * cos + x1 * sin
        return np.concatenate([out1, out2], axis=2).reshape(batch, hidden_dim)

    def _rope_cos_sin(self, pos: int):
        half = self.cfg.head_dim // 2
        inv_freq = 1.0 / (self.cfg.rope_theta ** (np.arange(0, half, dtype=np.float32) / half))
        angles = pos * inv_freq
        return np.cos(angles), np.sin(angles)
