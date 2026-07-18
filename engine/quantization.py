"""
Weight quantization.

Two schemes are provided:

  * `WeightQuantizer`   -- group-wise *uniform* asymmetric INT-N (GPTQ/AWQ-style).
                           Supports 2, 4, or 8-bit codes (`EngineConfig.quant_bits`).
                           Fast, simple, good default.
  * `NF4Quantizer`      -- group-wise *NormalFloat4* (QLoRA-style). Weights in
                           a transformer are close to zero-centered Gaussian,
                           so allocating the 16 representable levels at the
                           quantiles of a standard normal (instead of
                           uniformly) gives measurably lower reconstruction
                           error per bit than uniform INT4 for typical weight
                           distributions.

Layout (both schemes)
----------------------
A weight matrix `w` of shape (out_features, in_features) is quantized in
groups of `group_size` columns (in_features axis), each group carrying its
own per-row (per-output-channel) scale (+ zero-point for uniform INT-N).
`values_per_byte = 8 // bits` codes are packed into each output byte
(low-bits-first) -- this is exactly what `kernels.fused_int4_gemv` /
`fused_int4_gemm_batch` and the CUDA kernels consume directly, so no
unpacking round-trip is needed on the hot path.

Both quantizers are fully vectorized (reshape + broadcast, no Python loop
over groups), so `quantize()` on a full weight matrix is a handful of NumPy
ops regardless of `n_groups`.
"""
from __future__ import annotations

import numpy as np

from .config import EngineConfig

# QLoRA NormalFloat4 code book: 16 values, information-theoretically optimal
# quantization levels for a zero-centered unit-normal distribution (Dettmers
# et al., 2023). Weights are close to Gaussian in practice, so placing more
# resolution near zero (where most probability mass sits) beats uniform INT4.
NF4_LEVELS = np.array([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=np.float32)


def _pack_bits(q: np.ndarray, bits: int) -> np.ndarray:
    """Pack `values_per_byte = 8 // bits` codes per byte, low-bits-first.

    q : uint8, shape (out_features, in_features), values in [0, 2**bits - 1]
    """
    vpb = 8 // bits
    out_features, in_features = q.shape
    if in_features % vpb != 0:
        raise ValueError(f"in_features must be divisible by {vpb} to pack {bits}-bit codes")
    packed = np.zeros((out_features, in_features // vpb), dtype=np.uint8)
    for j in range(vpb):
        packed |= (q[:, j::vpb] & ((1 << bits) - 1)).astype(np.uint8) << (bits * j)
    return packed


def _unpack_bits(packed: np.ndarray, in_features: int, bits: int) -> np.ndarray:
    vpb = 8 // bits
    q = np.empty((packed.shape[0], in_features), dtype=np.uint8)
    mask = (1 << bits) - 1
    for j in range(vpb):
        q[:, j::vpb] = (packed >> (bits * j)) & mask
    return q


def quantize_activations_batch(X: np.ndarray):
    """Per-row dynamic INT8 quantization for a batch of activation vectors.

    X : float32/float16, shape (batch, in_features)
    Returns (X_int8, scales) where scales has shape (batch,) and
    X_int8[i] * scales[i] ~= X[i]. Fully vectorized (no Python loop over rows);
    used by the batched prefill / continuous-batching decode path.
    """
    X32 = X.astype(np.float32)
    amax = np.max(np.abs(X32), axis=1) + 1e-8
    scales = (amax / 127.0).astype(np.float32)
    Xq = np.clip(np.round(X32 / scales[:, None]), -127, 127).astype(np.int8)
    return Xq, scales


def get_quantizer(config: EngineConfig):
    """Factory: build the quantizer selected by `config.quant_scheme`."""
    if config.quant_scheme == "nf4":
        return NF4Quantizer(config)
    return WeightQuantizer(config)


class WeightQuantizer:
    """Group-wise asymmetric uniform INT-N quantization (N = 2, 4, or 8 bits)."""

    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.group_size = config.group_size
        self.bits = config.quant_bits
        if self.bits not in (2, 4, 8):
            raise ValueError("WeightQuantizer supports quant_bits in {2, 4, 8}")
        self.qmax = 2 ** self.bits - 1
        self.values_per_byte = 8 // self.bits

    # ------------------------------------------------------------------ #
    def quantize(self, w: np.ndarray):
        """Quantize a (out_features, in_features) fp16/fp32 matrix.

        Returns
        -------
        qw_packed : uint8 array, shape (out_features, in_features // values_per_byte)
        scales    : float32 array, shape (out_features, n_groups)
        zeros     : float32 array, shape (out_features, n_groups)
        """
        if w.ndim != 2:
            raise ValueError("quantize expects a 2D (out_features, in_features) matrix")
        out_features, in_features = w.shape
        if in_features % self.group_size != 0:
            raise ValueError("in_features must be divisible by group_size")
        if in_features % self.values_per_byte != 0:
            raise ValueError(
                f"in_features must be divisible by {self.values_per_byte} "
                f"to pack {self.bits}-bit values into bytes"
            )

        n_groups = in_features // self.group_size
        # Fully vectorized: reshape into (out, n_groups, group_size) and
        # reduce over the last axis -- no Python loop over groups.
        w3 = w.astype(np.float32).reshape(out_features, n_groups, self.group_size)
        wmin = w3.min(axis=2)                       # (out, n_groups)
        wmax = w3.max(axis=2)
        scale = (wmax - wmin) / self.qmax
        scale = np.where(scale == 0, 1e-8, scale)
        zero = -wmin / scale

        q3 = np.clip(np.round(w3 / scale[:, :, None] + zero[:, :, None]), 0, self.qmax)
        q_full = q3.astype(np.uint8).reshape(out_features, in_features)

        qw_packed = _pack_bits(q_full, self.bits)
        return qw_packed, scale.astype(np.float32), zero.astype(np.float32)

    # ------------------------------------------------------------------ #
    def dequantize_group(self, qw_packed: np.ndarray, scales: np.ndarray,
                          zeros: np.ndarray, g: int) -> np.ndarray:
        """Dequantize a single group `g` back to float32, shape (out, group_size)."""
        bytes_per_group = self.group_size // self.values_per_byte
        packed_g = qw_packed[:, g * bytes_per_group:(g + 1) * bytes_per_group]
        q = _unpack_bits(packed_g, self.group_size, self.bits).astype(np.float32)
        scale = scales[:, g:g + 1]
        zero = zeros[:, g:g + 1]
        return (q - zero) * scale

    def dequantize(self, qw_packed: np.ndarray, scales: np.ndarray, zeros: np.ndarray) -> np.ndarray:
        """Dequantize the full matrix back to float32, shape (out, in_features).
        Fully vectorized: unpacks all groups at once via reshape."""
        out_features, packed_in = qw_packed.shape
        in_features = packed_in * self.values_per_byte
        n_groups = scales.shape[1]
        q = _unpack_bits(qw_packed, in_features, self.bits).astype(np.float32)
        q3 = q.reshape(out_features, n_groups, self.group_size)
        w3 = (q3 - zeros[:, :, None]) * scales[:, :, None]
        return w3.reshape(out_features, in_features)

    def memory_bytes(self, out_features: int, in_features: int) -> int:
        """Total bytes for weights + fp32 scale/zero metadata -- useful for
        the memory-budget comparisons the README table cites."""
        n_groups = in_features // self.group_size
        weight_bytes = out_features * in_features // self.values_per_byte
        meta_bytes = out_features * n_groups * 4 * 2   # scale + zero, fp32
        return weight_bytes + meta_bytes


class NF4Quantizer:
    """Group-wise NormalFloat4 (QLoRA-style) quantization.

    Each group is scaled by its absmax into [-1, 1], then every value is
    snapped to the nearest of the 16 `NF4_LEVELS` (a 4-bit code). Unlike
    uniform INT4, resolution is concentrated near zero, which matches the
    approximately-Gaussian shape of trained transformer weights and
    typically yields lower reconstruction MSE than `WeightQuantizer` at the
    same bit width -- at the cost of a lookup-table dequantization instead of
    a single affine transform. No zero-point: NF4 codes are inherently
    zero-centered, so only a per-group scale is stored.
    """

    def __init__(self, config: EngineConfig):
        self.cfg = config
        self.group_size = config.group_size
        self.levels = NF4_LEVELS
        self.values_per_byte = 2

    def quantize(self, w: np.ndarray):
        """Returns (qw_packed uint8, scales float32 (out, n_groups), None).
        The trailing None keeps the same 3-tuple contract as
        `WeightQuantizer.quantize` (no zero-point needed: NF4 codes are
        inherently zero-centered)."""
        if w.ndim != 2:
            raise ValueError("quantize expects a 2D (out_features, in_features) matrix")
        out_features, in_features = w.shape
        if in_features % self.group_size != 0:
            raise ValueError("in_features must be divisible by group_size")
        if in_features % 2 != 0:
            raise ValueError("in_features must be even to pack 2 codes per byte")

        n_groups = in_features // self.group_size
        w3 = w.astype(np.float32).reshape(out_features, n_groups, self.group_size)
        absmax = np.max(np.abs(w3), axis=2)
        absmax = np.where(absmax == 0, 1e-8, absmax)
        w_norm = w3 / absmax[:, :, None]             # in [-1, 1]

        # Nearest-level lookup, vectorized: a full (out, n_groups, group_size, 16)
        # distance tensor would be wasteful for big models, so we search via
        # searchsorted on the (monotonic) level table instead.
        flat = w_norm.reshape(-1)
        idx_hi = np.clip(np.searchsorted(self.levels, flat), 1, len(self.levels) - 1)
        idx_lo = idx_hi - 1
        dist_hi = np.abs(self.levels[idx_hi] - flat)
        dist_lo = np.abs(self.levels[idx_lo] - flat)
        codes = np.where(dist_lo <= dist_hi, idx_lo, idx_hi).astype(np.uint8)
        q_full = codes.reshape(out_features, in_features)

        qw_packed = _pack_bits(q_full, bits=4)
        return qw_packed, absmax.astype(np.float32), None

    def dequantize(self, qw_packed: np.ndarray, scales: np.ndarray, zeros: np.ndarray = None) -> np.ndarray:
        """`zeros` is accepted (and ignored) purely so call sites can treat
        WeightQuantizer and NF4Quantizer interchangeably."""
        out_features, packed_in = qw_packed.shape
        in_features = packed_in * 2
        n_groups = scales.shape[1]
        codes = _unpack_bits(qw_packed, in_features, bits=4)
        q3 = codes.reshape(out_features, n_groups, self.group_size)
        w3 = self.levels[q3] * scales[:, :, None]
        return w3.reshape(out_features, in_features)
