"""
Minimal, dependency-free reader for the GGUF format (llama.cpp's model
container format, spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md).

Supports:
  * The full binary header (magic, version, tensor/KV counts)
  * All GGUF metadata value types (ints, floats, bool, string, array)
  * Tensor info parsing (name, shape, ggml type, data offset) for every tensor
  * Raw data extraction for F32, F16, and the 8/16/32/64-bit int ggml types

Deliberately NOT supported: the k-quant / legacy block-quantized ggml types
(Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0, etc). Correctly dequantizing those
requires exactly reproducing many block-format-specific bit layouts, which
is easy to get subtly wrong without a reference file to validate against
(and this environment has no network access to fetch one). `load_tensor`
raises a clear `NotImplementedError` naming the unsupported type rather than
silently producing wrong numbers -- see README "Status & scope". F32/F16
GGUF exports (e.g. `--outtype f16` in llama.cpp's converter) work today;
add a k-quant dequantizer here to extend coverage.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

GGUF_MAGIC = b"GGUF"

# ggml_type enum -> (numpy dtype or None if block-quantized, bytes-per-element or None)
_GGML_TYPE_F32 = 0
_GGML_TYPE_F16 = 1
_GGML_TYPE_I8 = 24
_GGML_TYPE_I16 = 25
_GGML_TYPE_I32 = 26
_GGML_TYPE_I64 = 27
_GGML_TYPE_F64 = 28

_SIMPLE_TYPES = {
    _GGML_TYPE_F32: (np.float32, 4),
    _GGML_TYPE_F16: (np.float16, 2),
    _GGML_TYPE_I8: (np.int8, 1),
    _GGML_TYPE_I16: (np.int16, 2),
    _GGML_TYPE_I32: (np.int32, 4),
    _GGML_TYPE_I64: (np.int64, 8),
    _GGML_TYPE_F64: (np.float64, 8),
}
_BLOCK_QUANT_NAMES = {
    2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K",
}

# GGUF metadata value type enum
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32 = 0, 1, 2, 3, 4, 5
_T_FLOAT32, _T_BOOL, _T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12

_SCALAR_STRUCTS = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_UINT64: "<Q",
    _T_INT64: "<q", _T_FLOAT64: "<d",
}


class _Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def read(self, n: int) -> bytes:
        chunk = self.data[self.pos:self.pos + n]
        self.pos += n
        return chunk

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def string(self) -> str:
        length = self.u64()
        return self.read(length).decode("utf-8")

    def value(self, vtype: int) -> Any:
        if vtype == _T_STRING:
            return self.string()
        if vtype == _T_BOOL:
            return struct.unpack("<B", self.read(1))[0] != 0
        if vtype == _T_ARRAY:
            elem_type = self.i32()
            n = self.u64()
            return [self.value(elem_type) for _ in range(n)]
        fmt = _SCALAR_STRUCTS.get(vtype)
        if fmt is None:
            raise NotImplementedError(f"Unknown GGUF metadata value type {vtype}")
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size))[0]


@dataclass
class GGUFTensorInfo:
    name: str
    shape: Tuple[int, ...]
    ggml_type: int
    offset: int          # relative to the start of the tensor data section


@dataclass
class GGUFFile:
    version: int
    metadata: Dict[str, Any]
    tensor_infos: List[GGUFTensorInfo]
    _data: bytes           # full file bytes
    _data_section_start: int
    alignment: int

    def tensor_names(self) -> List[str]:
        return [t.name for t in self.tensor_infos]

    def load_tensor(self, name: str) -> np.ndarray:
        info = next((t for t in self.tensor_infos if t.name == name), None)
        if info is None:
            raise KeyError(f"No tensor named {name!r} in this GGUF file")

        if info.ggml_type in _SIMPLE_TYPES:
            np_dtype, elem_size = _SIMPLE_TYPES[info.ggml_type]
            count = int(np.prod(info.shape)) if info.shape else 1
            start = self._data_section_start + info.offset
            raw = self._data[start:start + count * elem_size]
            # GGUF stores dims fastest-varying first (like ggml); reverse to
            # get NumPy's C-order (slowest-varying first) row-major shape.
            arr = np.frombuffer(raw, dtype=np_dtype, count=count)
            return arr.reshape(tuple(reversed(info.shape))).copy()

        type_name = _BLOCK_QUANT_NAMES.get(info.ggml_type, f"ggml_type={info.ggml_type}")
        raise NotImplementedError(
            f"Tensor '{name}' uses block-quantized GGUF type {type_name}, which this "
            f"lightweight reader does not dequantize (see engine/loaders/gguf_io.py "
            f"module docstring). Re-export the checkpoint as F16/F32 GGUF, or extend "
            f"`load_tensor` with a {type_name} dequantizer."
        )


def load(path: str) -> GGUFFile:
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file (bad magic): {path}")

    r = _Reader(data)
    r.read(4)  # magic already checked
    version = r.i32()
    n_tensors = r.u64()
    n_kv = r.u64()

    metadata: Dict[str, Any] = {}
    for _ in range(n_kv):
        key = r.string()
        vtype = r.i32()
        metadata[key] = r.value(vtype)

    tensor_infos: List[GGUFTensorInfo] = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.i32()
        shape = tuple(r.u64() for _ in range(n_dims))
        ggml_type = r.i32()
        offset = r.u64()
        tensor_infos.append(GGUFTensorInfo(name, shape, ggml_type, offset))

    alignment = int(metadata.get("general.alignment", 32))
    data_section_start = r.pos
    remainder = data_section_start % alignment
    if remainder != 0:
        data_section_start += alignment - remainder

    return GGUFFile(version=version, metadata=metadata, tensor_infos=tensor_infos,
                     _data=data, _data_section_start=data_section_start, alignment=alignment)
