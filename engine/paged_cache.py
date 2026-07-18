"""
Block/page-based KV cache (a simplified, CPU/NumPy version of vLLM's
PagedAttention memory model).

`model.KVCache` is a fine per-sequence cache, but it grows an unbounded
Python list per sequence -- fine for one request at a time, wasteful once
you're juggling many concurrent sequences of different lengths, because
there's no way to reclaim or share memory at anything finer than "the whole
sequence is done."

`PagePool` + `PagedSequenceCache` split KV storage into fixed-size pages
drawn from a shared pool:

  * Pages are allocated on demand as a sequence grows, `page_size` tokens
    at a time, instead of one array per sequence sized for `max_seq_len`.
  * When a sequence is truncated (e.g. after a rejected speculative-decoding
    round) or finishes, its pages go back to the pool's free list and are
    immediately available to the *next* sequence -- no cross-sequence
    fragmentation.
  * `PagedSequenceCache` implements the exact same duck-typed interface as
    `model.KVCache` (`append`, `get`, `length`, `truncate`), so it's a
    drop-in replacement anywhere a `KVCache` is used -- including inside
    `SpeculativeDecoder` and `Scheduler` -- controlled by
    `EngineConfig.use_paged_cache`.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class PagePool:
    """Shared free-list allocator for fixed-size KV pages."""

    def __init__(self, page_size: int = 16):
        self.page_size = page_size
        self._free_ids: List[int] = []
        self._next_id = 0
        self.total_allocated = 0
        self.peak_allocated = 0

    def alloc(self) -> int:
        if self._free_ids:
            pid = self._free_ids.pop()
        else:
            pid = self._next_id
            self._next_id += 1
        self.total_allocated += 1
        self.peak_allocated = max(self.peak_allocated, self.total_allocated)
        return pid

    def free(self, page_id: int):
        self._free_ids.append(page_id)
        self.total_allocated -= 1

    @property
    def n_pages_ever_created(self) -> int:
        return self._next_id


class PagedSequenceCache:
    """A single sequence's KV cache, backed by pages drawn from a shared `PagePool`.

    Same interface as `model.KVCache`: `append(layer, k, v)`, `get(layer)`,
    `length(layer)`, `truncate(layer, length)`.
    """

    def __init__(self, n_layers: int, pool: PagePool):
        self.n_layers = n_layers
        self.pool = pool
        self.page_size = pool.page_size
        self._block_tables: List[List[int]] = [[] for _ in range(n_layers)]
        self._pages: Dict[Tuple[int, int], Dict[str, List[np.ndarray]]] = {}
        self._lengths: List[int] = [0] * n_layers

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray):
        bt = self._block_tables[layer_idx]
        length = self._lengths[layer_idx]
        slot = length % self.page_size
        if slot == 0:
            pid = self.pool.alloc()
            bt.append(pid)
            self._pages[(layer_idx, pid)] = {"k": [], "v": []}
        page = self._pages[(layer_idx, bt[-1])]
        page["k"].append(k)
        page["v"].append(v)
        self._lengths[layer_idx] += 1

    def get(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        ks: List[np.ndarray] = []
        vs: List[np.ndarray] = []
        for pid in self._block_tables[layer_idx]:
            page = self._pages[(layer_idx, pid)]
            ks.extend(page["k"])
            vs.extend(page["v"])
        return np.stack(ks), np.stack(vs)

    def length(self, layer_idx: int) -> int:
        return self._lengths[layer_idx]

    def truncate(self, layer_idx: int, length: int):
        """Drop whole pages beyond `length` (returning them to the pool) and
        trim the new last page down to its remaining valid slots."""
        keep_pages = (length + self.page_size - 1) // self.page_size if length > 0 else 0
        bt = self._block_tables[layer_idx]
        while len(bt) > keep_pages:
            pid = bt.pop()
            self.pool.free(pid)
            del self._pages[(layer_idx, pid)]
        if bt:
            remainder = length - (len(bt) - 1) * self.page_size
            page = self._pages[(layer_idx, bt[-1])]
            page["k"] = page["k"][:remainder]
            page["v"] = page["v"][:remainder]
        self._lengths[layer_idx] = length

    def free(self):
        """Release every page held by this sequence back to the shared pool."""
        for layer_idx in range(self.n_layers):
            for pid in self._block_tables[layer_idx]:
                self.pool.free(pid)
                del self._pages[(layer_idx, pid)]
            self._block_tables[layer_idx] = []
        self._lengths = [0] * self.n_layers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.free()
