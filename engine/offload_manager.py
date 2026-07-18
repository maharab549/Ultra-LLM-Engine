"""
Async CPU <-> GPU layer offload.

Only the first `gpu_layers` transformer layers stay permanently resident on
the "GPU" (in this CPU-only reference implementation, "GPU" residency is
simulated as a plain in-memory dict; on real hardware this would be a CuPy
array pinned to device memory). The remaining "cold" layers live in
pinned-ish host memory (a regular dict here) and are pulled onto the GPU
tier just-in-time.

The key idea that makes offload cheap is **prefetch**: while layer `i` is
executing on the GPU, a background thread is already copying layer `i + 1`
(and up to `prefetch_depth` layers ahead) from host memory, so the PCIe
transfer is hidden behind compute instead of stalling the critical path.
"""
from __future__ import annotations

import threading
import queue
from typing import Any, Dict, Optional


class OffloadManager:
    def __init__(self, layers: Dict[int, Any], gpu_layers: int = 4, prefetch_depth: int = 2):
        """
        layers : mapping layer_index -> layer weights/object (lives in "host" memory)
        gpu_layers : number of layers that are always resident on GPU (layers 0..gpu_layers-1)
        prefetch_depth : how many layers ahead of the current cursor to prefetch
        """
        self._host_layers = layers
        self.n_layers = len(layers)
        self.gpu_layers = min(gpu_layers, self.n_layers)
        self.prefetch_depth = prefetch_depth

        # "GPU-resident" tier: hot layers are here permanently, cold layers
        # are placed here transiently once fetched.
        self._gpu_tier: Dict[int, Any] = {i: layers[i] for i in range(self.gpu_layers)}
        self._gpu_lock = threading.Lock()

        self._prefetch_queue: "queue.Queue[Optional[int]]" = queue.Queue()
        self._prefetch_events: Dict[int, threading.Event] = {}
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._prefetch_worker, daemon=True)
        self._worker.start()

        self.stats = {"host_to_gpu_copies": 0, "cache_hits": 0, "cache_misses": 0}

    # ------------------------------------------------------------------ #
    def _prefetch_worker(self):
        while not self._stop.is_set():
            try:
                idx = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if idx is None:
                continue
            self._materialize(idx)
            evt = self._prefetch_events.get(idx)
            if evt is not None:
                evt.set()

    def _materialize(self, idx: int):
        """Actually copy a layer from host memory into the GPU tier."""
        with self._gpu_lock:
            if idx not in self._gpu_tier:
                self._gpu_tier[idx] = self._host_layers[idx]  # simulated PCIe copy
                self.stats["host_to_gpu_copies"] += 1

    # ------------------------------------------------------------------ #
    def prefetch(self, idx: int):
        """Kick off an async copy of layer `idx` if it's cold and not already queued."""
        if idx < 0 or idx >= self.n_layers or idx < self.gpu_layers:
            return
        with self._gpu_lock:
            if idx in self._gpu_tier or idx in self._prefetch_events:
                return
            self._prefetch_events[idx] = threading.Event()
        self._prefetch_queue.put(idx)

    def get_layer(self, idx: int, block: bool = True) -> Any:
        """Fetch layer `idx`, blocking until its async copy finishes if needed,
        then immediately kick off prefetch for the upcoming window."""
        if idx in self._gpu_tier:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1
            evt = self._prefetch_events.get(idx)
            if evt is None:
                # Not prefetched ahead of time (e.g. cold start) -> fetch synchronously.
                self._materialize(idx)
            elif block:
                evt.wait()

        # Slide the prefetch window forward.
        for d in range(1, self.prefetch_depth + 1):
            self.prefetch(idx + d)

        # Evict cold layers that have fallen behind the window to bound memory.
        self._evict_stale(idx)
        return self._gpu_tier[idx]

    def _evict_stale(self, current_idx: int):
        with self._gpu_lock:
            stale = [
                i for i in list(self._gpu_tier.keys())
                if i >= self.gpu_layers and i < current_idx
            ]
            for i in stale:
                del self._gpu_tier[i]
                self._prefetch_events.pop(i, None)

    def shutdown(self):
        self._stop.set()
        self._worker.join(timeout=1.0)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
