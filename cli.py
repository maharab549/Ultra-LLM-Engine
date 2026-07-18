#!/usr/bin/env python3
"""
CLI entry point.

    python cli.py --prompt "Explain speculative decoding" --speculate 5

By default this runs with a small *randomly-initialized* model so the whole
pipeline (tokenize -> offload -> quantized kernels -> speculative decode ->
detokenize) is runnable and inspectable with zero external weight files.

Pass --checkpoint to load a real .safetensors or .gguf checkpoint (shape
inferred from --checkpoint-format / the GGUF file's own metadata via
`engine.loaders.checkpoint.config_from_gguf_metadata`, or specify
--hidden-dim etc. by hand for a .safetensors checkpoint).
"""
from __future__ import annotations

import argparse
import time

from engine.config import EngineConfig
from engine.kernels import FusedKernels
from engine.loaders import checkpoint as ckpt
from engine.model import QuantizedTransformer
from engine.scheduler import Scheduler
from engine.tokenizer import BPETokenizer


def build_config(args) -> EngineConfig:
    return EngineConfig(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        group_size=args.group_size,
        quant_bits=args.quant_bits,
        quant_scheme=args.quant_scheme,
        gpu_layers=args.gpu_layers,
        speculate_k=args.speculate,
        adaptive_speculation=args.adaptive_k,
        use_paged_cache=args.paged_cache,
        page_size=args.page_size,
        seed=args.seed,
    )


def load_target_model(args, cfg: EngineConfig, kernels: FusedKernels) -> QuantizedTransformer:
    if args.checkpoint is None:
        print("[cli] no --checkpoint given -- using a random model of the configured shape.")
        return QuantizedTransformer.from_random(cfg, kernels)

    fmt = args.checkpoint_format
    if fmt == "auto":
        fmt = "gguf" if args.checkpoint.endswith(".gguf") else "safetensors"

    print(f"[cli] loading {fmt} checkpoint: {args.checkpoint}")
    if fmt == "gguf":
        return ckpt.from_gguf(args.checkpoint, cfg, kernels)
    return ckpt.from_safetensors(args.checkpoint, cfg, kernels)


def main():
    ap = argparse.ArgumentParser(description="Ultra-Lightweight LLM Inference Engine")
    ap.add_argument("--checkpoint", type=str, default=None,
                     help="Path to a .safetensors or .gguf target-model checkpoint")
    ap.add_argument("--checkpoint-format", choices=["auto", "safetensors", "gguf"], default="auto")
    ap.add_argument("--draft-checkpoint", type=str, default=None,
                     help="Path to a draft-model checkpoint (only used with --draft-mode model)")

    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--batch-prompts", type=str, default=None,
                     help="'|'-separated list of prompts to decode together via forward_batch_step")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)

    ap.add_argument("--gpu-layers", type=int, default=4)
    ap.add_argument("--paged-cache", action="store_true", help="Use the block/page-based KV cache")
    ap.add_argument("--page-size", type=int, default=16)

    ap.add_argument("--draft-mode", choices=["model", "ngram", "lookup", "greedy"], default="greedy")
    ap.add_argument("--speculate", type=int, default=5, help="Draft tokens per round (or initial k if --adaptive-k)")
    ap.add_argument("--adaptive-k", action="store_true", help="Grow/shrink speculate_k by rolling accept rate")
    ap.add_argument("--ngram-n", type=int, default=3)
    ap.add_argument("--lookup-n", type=int, default=3)

    ap.add_argument("--quant-bits", type=int, choices=[2, 4, 8], default=4)
    ap.add_argument("--quant-scheme", choices=["int", "nf4"], default="int")

    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.prompt and not args.batch_prompts:
        ap.error("pass --prompt \"...\" or --batch-prompts \"a|b|c\"")

    cfg = build_config(args)
    kernels = FusedKernels(cfg)
    print(f"[cli] kernel backend : {'CUDA' if kernels.using_cuda else 'NumPy (CPU fallback)'}")
    print(f"[cli] quantization   : {cfg.quant_bits}-bit {cfg.quant_scheme}, group_size={cfg.group_size}")

    tokenizer = BPETokenizer()
    seed_corpus = [args.prompt or ""] + (args.batch_prompts or "").split("|") + \
        ["the quick brown fox jumps over the lazy dog"]
    tokenizer.train([s for s in seed_corpus if s], num_merges=200)

    target_model = load_target_model(args, cfg, kernels)

    draft_model = None
    if args.draft_mode == "model":
        draft_cfg = EngineConfig(
            hidden_dim=max(64, cfg.hidden_dim // 4), n_layers=max(1, cfg.n_layers // 4),
            n_heads=max(1, cfg.n_heads // 2), vocab_size=cfg.vocab_size,
            group_size=min(cfg.group_size, max(64, cfg.hidden_dim // 4)), seed=cfg.seed,
        )
        draft_kernels = FusedKernels(draft_cfg)
        if args.draft_checkpoint:
            print(f"[cli] loading draft checkpoint: {args.draft_checkpoint}")
            fmt = "gguf" if args.draft_checkpoint.endswith(".gguf") else "safetensors"
            draft_model = (ckpt.from_gguf(args.draft_checkpoint, draft_cfg, draft_kernels) if fmt == "gguf"
                            else ckpt.from_safetensors(args.draft_checkpoint, draft_cfg, draft_kernels))
        else:
            draft_model = QuantizedTransformer.from_random(draft_cfg, draft_kernels)

    with Scheduler(cfg, tokenizer, target_model, draft_model, draft_mode=args.draft_mode,
                   ngram_n=args.ngram_n, lookup_n=args.lookup_n) as sched:
        t0 = time.time()
        if args.batch_prompts:
            prompts = args.batch_prompts.split("|")
            texts = sched.generate_batch(prompts, max_tokens=args.max_tokens, temperature=args.temperature)
            dt = time.time() - t0
            print(f"\n[cli] batch size     : {len(prompts)}")
            print(f"[cli] elapsed        : {dt:.2f}s ({len(prompts) * args.max_tokens / max(dt, 1e-6):.1f} tok/s aggregate)")
            for i, text in enumerate(texts):
                print(f"\n--- output[{i}] ---\n{text}\n")
        else:
            text = sched.generate(args.prompt, max_tokens=args.max_tokens, temperature=args.temperature)
            dt = time.time() - t0
            print(f"\n[cli] decoding mode  : {args.draft_mode}"
                  + (f" (adaptive-k, final k={sched.spec.k})" if sched.spec and sched.spec.adaptive else ""))
            print(f"[cli] gpu_layers     : {cfg.gpu_layers}/{cfg.n_layers}")
            print(f"[cli] offload stats  : {sched.offload.stats}")
            if sched.spec is not None:
                print(f"[cli] accept rate    : {sched.spec.acceptance_rate:.2f}")
            elif sched.lightweight_spec is not None:
                print(f"[cli] accept rate    : {sched.lightweight_spec.acceptance_rate:.2f}")
            print(f"[cli] elapsed        : {dt:.2f}s ({args.max_tokens / max(dt, 1e-6):.1f} tok/s)")
            print(f"\n--- output ---\n{text}\n")


if __name__ == "__main__":
    main()
