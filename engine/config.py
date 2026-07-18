"""Central configuration object shared by every engine component."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EngineConfig:
    # --- model shape -------------------------------------------------
    hidden_dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None       # None -> defaults to n_heads (MHA)
    ffn_dim: Optional[int] = None          # None -> defaults to 4 * hidden_dim
    vocab_size: int = 32000
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5

    # --- quantization --------------------------------------------------
    group_size: int = 128                  # weights per quantization group
    quant_bits: int = 4                    # 2, 4, or 8-bit weights
    quant_scheme: str = "int"              # "int" (uniform GPTQ-style) or "nf4" (QLoRA NormalFloat4)
    act_bits: int = 8                      # INT8 activations

    # --- offload ---------------------------------------------------------
    gpu_layers: int = 4                    # how many layers stay resident on GPU
    prefetch_depth: int = 2                # how many layers to prefetch ahead
    pin_memory: bool = True

    # --- KV cache ----------------------------------------------------------
    use_paged_cache: bool = False          # block/page-based KV cache (see engine.paged_cache)
    page_size: int = 16                    # tokens per page when use_paged_cache=True

    # --- speculative decoding -------------------------------------------
    speculate_k: int = 5                   # draft tokens proposed per round (or initial value if adaptive)
    draft_temperature: float = 1.0
    target_temperature: float = 1.0
    adaptive_speculation: bool = False     # grow/shrink k based on rolling accept rate
    min_speculate_k: int = 1
    max_speculate_k: int = 16

    # --- misc ------------------------------------------------------------
    seed: int = 0
    dtype: str = "float16"

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.ffn_dim is None:
            self.ffn_dim = 4 * self.hidden_dim
        if self.hidden_dim % self.n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        if self.hidden_dim % self.group_size != 0:
            raise ValueError("hidden_dim must be divisible by group_size")
        if self.quant_bits not in (2, 4, 8):
            raise ValueError("quant_bits must be one of 2, 4, 8")
        if self.quant_scheme not in ("int", "nf4"):
            raise ValueError("quant_scheme must be 'int' or 'nf4'")
        if self.quant_scheme == "nf4" and self.quant_bits != 4:
            raise ValueError("quant_scheme='nf4' requires quant_bits == 4 (NF4 is a 4-bit code)")
        if self.min_speculate_k < 1 or self.max_speculate_k < self.min_speculate_k:
            raise ValueError("require 1 <= min_speculate_k <= max_speculate_k")

    @property
    def head_dim(self) -> int:
        return self.hidden_dim // self.n_heads
