.PHONY: all cuda test clean run

NVCC ?= nvcc
CUDA_ARCH ?= sm_80
PY ?= python3

all: cuda

# CUDA kernels are JIT-compiled at runtime by CuPy's RawKernel (see
# engine/kernels.py), so this target is optional -- it just verifies the
# sources compile standalone (useful in CI on a GPU runner) and produces a
# .cubin you can inspect for register/occupancy tuning.
cuda:
	@command -v $(NVCC) >/dev/null 2>&1 || { \
		echo "nvcc not found -- skipping CUDA build. The engine will run on"; \
		echo "the pure-NumPy fallback kernels (see engine/kernels.py)."; \
		exit 0; \
	}
	$(NVCC) -arch=$(CUDA_ARCH) -cubin -o cuda/fused_int4_gemv.cubin cuda/fused_int4_gemv.cu
	$(NVCC) -arch=$(CUDA_ARCH) -cubin -o cuda/fused_rms_norm_rope.cubin cuda/fused_rms_norm_rope.cu
	@echo "CUDA kernels compiled."

test:
	$(PY) -m pytest tests/ -v

run:
	$(PY) cli.py --prompt "Explain speculative decoding" --speculate 5 --max-tokens 32

clean:
	rm -f cuda/*.cubin cuda/*.ptx
	find . -name "__pycache__" -type d -exec rm -rf {} +
