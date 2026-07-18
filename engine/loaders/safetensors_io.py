"""
Minimal, dependency-free reader/writer for the .safetensors format.

File layout (see https://github.com/huggingface/safetensors, "Format" section):

    [8 bytes] N = little-endian uint64, length of the header in bytes
    [N bytes] UTF-8 JSON header:
        {
          "tensor_name": {"dtype": "F32", "shape": [d0, d1, ...],
                           "data_offsets": [start, end]},
          ...
          "__metadata__": {...}            # optional, arbitrary string map
        }
    [rest]   raw tensor bytes, each tensor's bytes at
             data[data_offsets[0] : data_offsets[1]] (offsets relative to the
             start of the data section, i.e. right after the header)

Implemented independently of the official `safetensors` PyPI package so this
engine has zero third-party dependencies even for checkpoint I/O (and so it
works in fully offline environments). Only the dtypes this engine actually
needs are supported: F64, F32, F16, BF16, I64, I32, I16, I8, U8, BOOL.
"""
from __future__ import annotations

import json
import struct
from typing import Dict, Optional, Tuple

import numpy as np

_DTYPE_MAP = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8,
    "U8": np.uint8, "BOOL": np.bool_,
    # BF16 has no native NumPy dtype; we upcast to float32 on load (see below).
    "BF16": np.uint16,
}
_REVERSE_DTYPE_MAP = {
    np.dtype("float64"): "F64", np.dtype("float32"): "F32", np.dtype("float16"): "F16",
    np.dtype("int64"): "I64", np.dtype("int32"): "I32", np.dtype("int16"): "I16",
    np.dtype("int8"): "I8", np.dtype("uint8"): "U8", np.dtype("bool"): "BOOL",
}


def _bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """bfloat16 is the top 16 bits of an IEEE-754 float32; upcast by
    left-shifting into a 32-bit word."""
    as_u32 = raw_u16.astype(np.uint32) << 16
    return as_u32.view(np.float32)


def load_file(path: str) -> Dict[str, np.ndarray]:
    """Load every tensor in a .safetensors file into a dict of NumPy arrays."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
        data_start = 8 + header_len
        data = f.read()

    tensors: Dict[str, np.ndarray] = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype_str = info["dtype"]
        shape = tuple(info["shape"])
        start, end = info["data_offsets"]
        raw = data[start:end]

        if dtype_str == "BF16":
            arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
            tensors[name] = _bf16_to_f32(arr)
        else:
            np_dtype = _DTYPE_MAP.get(dtype_str)
            if np_dtype is None:
                raise NotImplementedError(f"safetensors dtype '{dtype_str}' not supported")
            tensors[name] = np.frombuffer(raw, dtype=np_dtype).reshape(shape).copy()
    return tensors


def load_metadata(path: str) -> Optional[Dict[str, str]]:
    """Read just the optional `__metadata__` string map without loading tensors."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return header.get("__metadata__")


def save_file(tensors: Dict[str, np.ndarray], path: str, metadata: Optional[Dict[str, str]] = None):
    """Write a dict of NumPy arrays out as a .safetensors file."""
    header: Dict[str, dict] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}

    offset = 0
    ordered_names = list(tensors.keys())
    buffers = []
    for name in ordered_names:
        arr = np.ascontiguousarray(tensors[name])
        dtype_key = _REVERSE_DTYPE_MAP.get(arr.dtype)
        if dtype_key is None:
            raise NotImplementedError(f"Cannot save NumPy dtype {arr.dtype} to safetensors")
        nbytes = arr.nbytes
        header[name] = {
            "dtype": dtype_key,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        buffers.append(arr.tobytes())
        offset += nbytes

    header_bytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for buf in buffers:
            f.write(buf)


def tensor_names(path: str) -> Tuple[str, ...]:
    """List tensor names in a .safetensors file without loading any data."""
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len).decode("utf-8"))
    return tuple(k for k in header.keys() if k != "__metadata__")
