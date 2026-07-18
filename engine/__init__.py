"""
Ultra-Lightweight LLM Inference Engine
---------------------------------------
A compact, dependency-light engine demonstrating three techniques that make
large models runnable on small GPUs:

    1. INT4 group quantization with a fused (dequant + GEMV) kernel
    2. Async CPU <-> "GPU" layer offload with background prefetch
    3. Speculative decoding (small draft model + big target model)

Every module has a pure-NumPy fallback so the whole pipeline runs and is
testable on a machine with no GPU/CUDA toolchain. When CuPy + nvcc are
available, `engine.kernels` will transparently use the compiled CUDA kernels
in `cuda/` instead of the NumPy fallback (see kernels.py for the switch).
"""

from .config import EngineConfig

__all__ = ["EngineConfig"]
__version__ = "0.2.0"
