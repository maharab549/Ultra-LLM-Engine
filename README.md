# Ultra LLM Engine

> **Caption:** A compact, from-scratch LLM inference engine for learning and
> experimenting with the systems ideas behind modern local model serving.

**GitHub description:** Dependency-light NumPy/CUDA reference engine for
quantized LLM inference with offload, paged KV cache, batched decode,
checkpoint loading, and speculative decoding.

Ultra LLM Engine is a small but complete decoder-only transformer runtime. It
is built to make the core mechanics of efficient LLM inference readable:
quantized weights, fused kernels, layer offload, KV-cache management, batched
generation, checkpoint loading, byte-level tokenization, and speculative
decoding all live in plain Python modules with focused tests.

The project runs out of the box on CPU with NumPy. If CuPy and a CUDA toolchain
are available, the engine can also try the CUDA fast path for selected fused
kernels. No model checkpoint is required for the demo path: the CLI can create
a tiny random model so the whole pipeline can be exercised end to end.

## Why this project exists

Most production LLM inference stacks are fast because they combine many ideas
at once: low-bit weights, careful memory movement, efficient cache layouts,
continuous or batched decoding, and draft-token verification. Those systems are
powerful, but they can be hard to study because the implementation is spread
across large C++/CUDA codebases.

This repository keeps the same ideas in a deliberately compact form. It is not
trying to replace llama.cpp, vLLM, TensorRT-LLM, or production serving systems.
It is a practical reference engine: small enough to read, real enough to run,
and tested enough to modify with confidence.

## Highlights

| Area | What is included |
| --- | --- |
| Quantization | Group-wise INT-2, INT-4, INT-8, and QLoRA-style NF4 weight quantization |
| Fused kernels | Fused dequant + GEMV/GEMM and fused RMSNorm + RoPE, with NumPy fallback and optional CUDA sources |
| Offload | Background layer prefetching through `OffloadManager` to model CPU/GPU tiering |
| KV cache | Plain per-sequence cache plus a reusable paged cache inspired by PagedAttention |
| Decoding | Greedy generation, neural speculative decoding, n-gram drafting, and prompt-lookup drafting |
| Batching | Multi-prompt decode that batches the expensive linear-layer work |
| Checkpoints | Pure-Python `.safetensors` loading and GGUF metadata/tensor loading for F16/F32 tensors |
| Tokenizer | Dependency-free byte-level BPE tokenizer with lossless round trips |
| Testing | Focused pytest suite covering quantization, kernels, cache behavior, decoding, loaders, and CLI paths |

## Quick Start

```bash
pip install -r requirements.txt
python -m pytest tests/ -v
python cli.py --prompt "Explain speculative decoding" --draft-mode ngram --max-tokens 32
```

The default CLI path uses a randomly initialized model. That means you can test
the engine mechanics without downloading weights:

```bash
python cli.py \
    --prompt "Explain speculative decoding" \
    --hidden-dim 256 \
    --n-layers 4 \
    --n-heads 8 \
    --vocab-size 1024 \
    --draft-mode greedy \
    --max-tokens 32
```

Try a lightweight speculative drafter:

```bash
python cli.py \
    --prompt "The fastest inference engines are" \
    --draft-mode lookup \
    --speculate 5 \
    --paged-cache \
    --max-tokens 48
```

Run batched generation:

```bash
python cli.py \
    --batch-prompts "Explain quantization|Explain KV cache|Explain RoPE" \
    --draft-mode greedy \
    --max-tokens 24
```

Optionally sanity-compile the CUDA source files:

```bash
make cuda
```

The CUDA target is optional. If CUDA/CuPy is unavailable, the engine continues
to run through the NumPy fallback implementation.

## Project Layout

```text
ultra_llm_engine/
├── cli.py
├── requirements.txt
├── Makefile
├── LICENSE
├── README.md
├── cuda/
│   ├── fused_int4_gemv.cu
│   └── fused_rms_norm_rope.cu
├── engine/
│   ├── __init__.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── quantization.py
│   ├── kernels.py
│   ├── offload_manager.py
│   ├── paged_cache.py
│   ├── draft_strategies.py
│   ├── speculative.py
│   ├── scheduler.py
│   ├── model.py
│   └── loaders/
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── safetensors_io.py
│       └── gguf_io.py
└── tests/
    ├── __init__.py
    └── test_engine.py
```

## How the Engine Works

```text
Prompt text
    |
    v
Byte-level BPE tokenizer
    |
    v
Scheduler
    |
    +-- prefill prompt tokens
    +-- fetch/prefetch target layers through OffloadManager
    +-- select decoding strategy
    +-- update plain or paged KV cache
    |
    v
QuantizedTransformer
    |
    +-- RMSNorm + RoPE
    +-- quantized Q/K/V/O projections
    +-- attention over cached keys and values
    +-- quantized MLP projections
    |
    v
Token ids -> decoded text
```

At each layer, the model applies RMSNorm, rotary position embeddings,
quantized attention projections, cached self-attention, and a gated MLP. The
hot path is intentionally explicit so each optimization is easy to inspect.

## Core Components

### Configuration

`engine/config.py` defines `EngineConfig`, the shared configuration object for
model dimensions, quantization settings, cache behavior, offload depth, and
speculative decoding parameters.

### Quantized Model

`engine/model.py` implements `QuantizedTransformer`, a compact GPT-style
decoder-only transformer. Linear layers are stored as quantized weights through
`QuantLinear`, while activations are dynamically quantized for the GEMV/GEMM
paths.

### Quantization

`engine/quantization.py` provides two quantization families:

| Scheme | Description |
| --- | --- |
| `int` | Group-wise asymmetric uniform INT-N quantization for 2, 4, or 8 bits |
| `nf4` | NormalFloat4 quantization with a QLoRA-style codebook |

Both quantizers pack multiple low-bit values into bytes and expose matching
dequantization helpers used by the fused kernel layer.

### Fused Kernels

`engine/kernels.py` exposes fused operations for:

- dequantized weight GEMV during single-token decode
- dequantized weight GEMM during batched decode
- RMSNorm followed by RoPE

The Python/NumPy implementation is always available. The `cuda/` directory
contains optional CUDA kernels for experimentation and standalone compile
checks.

### Offload Manager

`engine/offload_manager.py` simulates a hot GPU tier and a cold host tier.
When a layer is requested, the manager serves it from the hot tier if present,
materializes it if needed, and starts background prefetching for upcoming
layers.

### Paged KV Cache

`engine/paged_cache.py` implements a small page allocator for key/value cache
storage. `PagedSequenceCache` offers the same interface as the simple
`KVCache`, while `PagePool` lets pages be reclaimed and reused after sequence
truncation or completion.

### Speculative Decoding

`engine/speculative.py` supports:

- neural draft-model speculative decoding with accept/reject verification
- adaptive speculation windows
- lightweight speculative decoding with statistical drafters

`engine/draft_strategies.py` includes:

- `NGramDraft`, a count-based n-gram drafter
- `PromptLookupDraft`, a zero-parameter drafter that proposes repeated
  continuations from the current context

### Checkpoint Loading

`engine/loaders/` contains dependency-free checkpoint readers:

| File | Purpose |
| --- | --- |
| `safetensors_io.py` | Read/write simple `.safetensors` files |
| `gguf_io.py` | Read GGUF metadata and F16/F32/int tensors |
| `checkpoint.py` | Map HF/Llama-style tensor names into `QuantizedTransformer` |

GGUF k-quant formats such as `Q4_0`, `Q4_K`, `Q5_K`, and `Q6_K` are detected
but not dequantized. The loader raises a clear `NotImplementedError` instead
of silently producing incorrect tensor values.

## CLI Examples

Greedy decoding:

```bash
python cli.py --prompt "Hello from the engine" --draft-mode greedy --max-tokens 32
```

N-gram speculative decoding:

```bash
python cli.py --prompt "Repeatable patterns help" --draft-mode ngram --speculate 4
```

Prompt-lookup speculative decoding:

```bash
python cli.py --prompt "alpha beta gamma alpha beta" --draft-mode lookup --speculate 5
```

Neural draft-model mode with a random small draft model:

```bash
python cli.py \
    --prompt "Explain KV cache reuse" \
    --draft-mode model \
    --speculate 4 \
    --adaptive-k \
    --max-tokens 32
```

NF4 weights and paged cache:

```bash
python cli.py \
    --prompt "Describe low-bit inference" \
    --quant-scheme nf4 \
    --quant-bits 4 \
    --paged-cache \
    --max-tokens 32
```

Load a `.safetensors` checkpoint with explicit model shape:

```bash
python cli.py \
    --checkpoint /path/to/model.safetensors \
    --checkpoint-format safetensors \
    --hidden-dim 4096 \
    --n-layers 32 \
    --n-heads 32 \
    --vocab-size 32000 \
    --prompt "Explain attention"
```

Load a GGUF checkpoint:

```bash
python cli.py \
    --checkpoint /path/to/model.gguf \
    --checkpoint-format gguf \
    --hidden-dim 4096 \
    --n-layers 32 \
    --n-heads 32 \
    --vocab-size 32000 \
    --prompt "Explain quantized inference"
```

Use `python cli.py --help` for the full flag list.

## Testing

Run the full test suite:

```bash
python -m pytest tests/ -v
```

The tests cover:

- tokenizer encode/decode round trips
- INT-2, INT-4, INT-8, and NF4 quantization
- fused-kernel fallback numerics
- offload manager prefetch behavior
- greedy and speculative generation
- adaptive speculation bounds
- batched decode equivalence
- paged KV-cache behavior and page reuse
- safetensors and GGUF loader behavior

## Current Scope

Implemented:

- CPU-runnable inference path using NumPy
- optional CUDA kernel sources for selected fused operations
- random-model demo path
- real tensor loading from safetensors and supported GGUF tensors
- quantized transformer layers
- greedy, neural speculative, n-gram, and prompt-lookup decoding
- plain and paged KV-cache implementations
- batched multi-prompt decode path

Known limitations:

- This is a reference engine, not a production server.
- CUDA support is best-effort and optional.
- GGUF k-quant tensor dequantization is intentionally out of scope for now.
- Batched prefill currently left-pads shorter prompts without an attention mask.
- There is no flash attention, tensor parallelism, pipeline parallelism, or
  continuous batching scheduler.
- The default demo model is randomly initialized, so generated text is useful
  for exercising the pipeline, not for meaningful language quality.

## Roadmap Ideas

- Add GGUF k-quant dequantizers for common formats such as `Q4_0` and `Q8_0`.
- Add attention masking for variable-length batched prefill.
- Add a prefix-cache layer on top of the paged cache.
- Add benchmark scripts for comparing greedy, lookup, n-gram, and neural draft
  modes.
- Add a small documented checkpoint conversion example.
- Expand CUDA coverage for batched decode and cache-aware attention.

## Repository Metadata

Suggested short caption:

```text
From-scratch LLM inference engine with quantization, offload, paged KV cache,
batched decode, checkpoint loading, and speculative decoding.
```

Suggested GitHub topics:

```text
llm, inference, quantization, speculative-decoding, kv-cache, cuda, numpy,
transformer, gguf, safetensors
```

## License

MIT License. See `LICENSE` for details.

