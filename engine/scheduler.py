"""
Top-level orchestration loop: text -> tokens -> (prefill) -> generate -> text.

- During **prefill** (processing the prompt), the scheduler walks every
  layer sequentially and pulls each one through `OffloadManager.get_layer`,
  which transparently prefetches upcoming layers on a background thread.
- During **generation**, one of four strategies is used, selected by
  `draft_mode`:
    * `"model"`  -- neural draft model, `SpeculativeDecoder` (most general,
                    supports `adaptive_speculation` in `EngineConfig`)
    * `"ngram"`  -- statistical n-gram drafter, `LightweightSpeculativeDecoder`
                    + `NGramDraft` (no second model, "trained" instantly)
    * `"lookup"` -- prompt-lookup drafter, `LightweightSpeculativeDecoder`
                    + `PromptLookupDraft` (zero-parameter)
    * `"greedy"` -- plain token-by-token autoregressive decoding, no speculation
- `use_paged_cache=True` (in `EngineConfig`) swaps the per-sequence KV cache
  for a `PagedSequenceCache` backed by a shared `PagePool`, so memory used by
  a finished/truncated sequence is immediately reusable by the next one.
- `generate_batch` decodes several prompts together, batching every linear
  layer's GEMM across the batch dimension via `QuantizedTransformer.forward_batch_step`.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from .config import EngineConfig
from .draft_strategies import NGramDraft, PromptLookupDraft
from .model import KVCache, QuantizedTransformer
from .offload_manager import OffloadManager
from .paged_cache import PagePool, PagedSequenceCache
from .speculative import LightweightSpeculativeDecoder, SpeculativeDecoder
from .tokenizer import BPETokenizer

AnyCache = Union[KVCache, PagedSequenceCache]


class Scheduler:
    def __init__(self, cfg: EngineConfig, tokenizer: BPETokenizer,
                 target_model: QuantizedTransformer,
                 draft_model: Optional[QuantizedTransformer] = None,
                 draft_mode: Optional[str] = None, ngram_n: int = 3, lookup_n: int = 3,
                 ngram_corpus: Optional[List[List[int]]] = None):
        if draft_mode is None:
            draft_mode = "model" if draft_model is not None else "greedy"
        if draft_mode not in ("model", "ngram", "lookup", "greedy"):
            raise ValueError("draft_mode must be one of: model, ngram, lookup, greedy")
        if draft_mode == "model" and draft_model is None:
            raise ValueError("draft_mode='model' requires a draft_model")

        self.cfg = cfg
        self.tokenizer = tokenizer
        self.target = target_model
        self.draft = draft_model
        self.draft_mode = draft_mode

        layers = {i: layer for i, layer in enumerate(target_model.layers)}
        self.offload = OffloadManager(layers, gpu_layers=cfg.gpu_layers,
                                       prefetch_depth=cfg.prefetch_depth)

        self._page_pool = PagePool(page_size=cfg.page_size) if cfg.use_paged_cache else None

        self.spec: Optional[SpeculativeDecoder] = None
        self.lightweight_spec: Optional[LightweightSpeculativeDecoder] = None
        self.drafter = None

        if draft_mode == "model":
            self.spec = SpeculativeDecoder(
                draft_model, target_model, k=cfg.speculate_k,
                draft_temperature=cfg.draft_temperature, target_temperature=cfg.target_temperature,
                seed=cfg.seed, adaptive=cfg.adaptive_speculation,
                min_k=cfg.min_speculate_k, max_k=cfg.max_speculate_k,
            )
        elif draft_mode == "ngram":
            self.drafter = NGramDraft(n=ngram_n)
            if ngram_corpus:
                self.drafter.train(ngram_corpus)
            self.lightweight_spec = LightweightSpeculativeDecoder(
                self.drafter, target_model, k=cfg.speculate_k,
                target_temperature=cfg.target_temperature, seed=cfg.seed,
            )
        elif draft_mode == "lookup":
            self.drafter = PromptLookupDraft(lookup_n=lookup_n, max_k=cfg.speculate_k)
            self.lightweight_spec = LightweightSpeculativeDecoder(
                self.drafter, target_model, k=cfg.speculate_k,
                target_temperature=cfg.target_temperature, seed=cfg.seed,
            )

    # ------------------------------------------------------------------ #
    def _make_cache(self, n_layers: Optional[int] = None) -> AnyCache:
        n_layers = n_layers if n_layers is not None else self.cfg.n_layers
        if self._page_pool is not None:
            return PagedSequenceCache(n_layers, self._page_pool)
        return KVCache(n_layers)

    def _target_layer_provider(self, idx: int):
        return self.offload.get_layer(idx)

    def _truncate(self, cache: AnyCache, length: int, n_layers: Optional[int] = None):
        n_layers = n_layers if n_layers is not None else self.cfg.n_layers
        for i in range(n_layers):
            cache.truncate(i, length)

    # ------------------------------------------------------------------ #
    def _prefill(self, token_ids: List[int], cache: AnyCache, model: QuantizedTransformer,
                 use_offload: bool = False) -> np.ndarray:
        logits = None
        for pos, tok in enumerate(token_ids):
            provider = self._target_layer_provider if use_offload else None
            logits = model.forward_token(tok, pos, cache, layer_provider=provider)
        return logits

    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, max_tokens: int = 128,
                  temperature: Optional[float] = None) -> str:
        temperature = self.cfg.target_temperature if temperature is None else temperature
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True)
        if not prompt_ids:
            raise ValueError("Prompt tokenized to zero tokens")

        target_cache = self._make_cache()
        logits = self._prefill(prompt_ids, target_cache, self.target, use_offload=True)
        pos = len(prompt_ids) - 1
        last_token, _ = self.target.sample(logits, temperature)

        generated: List[int] = [last_token]

        if self.spec is not None:
            draft_cache = self._make_cache(self.draft.cfg.n_layers)
            self._prefill(prompt_ids, draft_cache, self.draft, use_offload=False)
            pos += 1
            while len(generated) < max_tokens:
                # cache_valid_length counts *newly appended* valid entries for
                # this round, not an absolute cache length -- both caches
                # already hold `pre_length` entries from the prefill/prior
                # rounds, so the real truncation target is pre_length + that.
                pre_length = target_cache.length(0)
                result = self.spec.generate_step(last_token, pos, draft_cache, target_cache)
                generated.extend(result.tokens)
                pos += len(result.tokens)
                target_len = pre_length + result.cache_valid_length
                self._truncate(draft_cache, target_len, self.draft.cfg.n_layers)
                self._truncate(target_cache, target_len)
                last_token = generated[-1]

        elif self.lightweight_spec is not None:
            if isinstance(self.drafter, NGramDraft):
                # Incorporate this prompt's own statistics -- cheap, and lets
                # the drafter pick up repeated phrasing within the prompt
                # itself even with no external corpus.
                self.drafter.train([prompt_ids])
            pos += 1
            history = list(prompt_ids) + generated
            while len(generated) < max_tokens:
                pre_length = target_cache.length(0)
                result = self.lightweight_spec.generate_step(history, pos, target_cache)
                generated.extend(result.tokens)
                history.extend(result.tokens)
                pos += len(result.tokens)
                self._truncate(target_cache, pre_length + result.cache_valid_length)

        else:  # greedy
            while len(generated) < max_tokens:
                pos += 1
                logits = self.target.forward_token(last_token, pos, target_cache,
                                                     layer_provider=self._target_layer_provider)
                last_token, _ = self.target.sample(logits, temperature)
                generated.append(last_token)

        return self._finalize(prompt_ids, generated)

    # ------------------------------------------------------------------ #
    def generate_batch(self, prompts: List[str], max_tokens: int = 64,
                        temperature: Optional[float] = None) -> List[str]:
        """Decode several prompts together, batching every linear layer's
        GEMM across the batch dimension (see `QuantizedTransformer.forward_batch_step`).

        Shorter prompts are left-padded with `<pad>` up to the longest
        prompt so every sequence reaches the same starting position before
        generation begins in lockstep. Note: this reference implementation
        does not apply an attention mask over the padding, so padded
        positions are attended to like real tokens -- a known simplification
        documented in the README, harmless for same-length or near-same-length
        batches and for the demo/random model, but worth masking properly
        before using this path for accuracy-critical batched serving.
        """
        temperature = self.cfg.target_temperature if temperature is None else temperature
        batch_ids = [self.tokenizer.encode(p, add_bos=True) for p in prompts]
        pad_id = self.tokenizer.token_to_id["<pad>"]
        max_len = max(len(ids) for ids in batch_ids)
        padded = [[pad_id] * (max_len - len(ids)) + ids for ids in batch_ids]
        batch = len(prompts)
        caches = [self._make_cache() for _ in range(batch)]

        logits = None
        for pos in range(max_len):
            toks = [padded[b][pos] for b in range(batch)]
            logits = self.target.forward_batch_step(toks, pos, caches,
                                                      layer_provider=self._target_layer_provider)

        last_tokens = [int(self.target.sample(logits[b], temperature)[0]) for b in range(batch)]
        generated = [[t] for t in last_tokens]
        pos = max_len - 1
        for _ in range(max_tokens - 1):
            pos += 1
            logits = self.target.forward_batch_step(last_tokens, pos, caches,
                                                      layer_provider=self._target_layer_provider)
            last_tokens = [int(self.target.sample(logits[b], temperature)[0]) for b in range(batch)]
            for b in range(batch):
                generated[b].append(last_tokens[b])

        return [self._finalize(batch_ids[b], generated[b]) for b in range(batch)]

    # ------------------------------------------------------------------ #
    def _finalize(self, prompt_ids: List[int], generated: List[int]) -> str:
        eos_id = self.tokenizer.token_to_id.get("<eos>")
        if eos_id in generated:
            generated = generated[:generated.index(eos_id)]
        return self.tokenizer.decode(prompt_ids + generated)

    def shutdown(self):
        self.offload.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.shutdown()
