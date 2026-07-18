"""
Lightweight "drafters" for speculative decoding that don't require training
or running a second neural network at all.

A neural draft model (see `engine.speculative.SpeculativeDecoder`) is the
most general option, but it costs a second set of weights, a second KV
cache, and offload traffic. Two well-established alternatives, both used in
production systems (llama.cpp, HF `transformers`' "prompt lookup decoding"),
trade generality for being essentially free:

  * `NGramDraft`      -- a classic n-gram language model. "Training" is just
                          counting n-gram frequencies over a corpus (no
                          gradients, no backprop) -- the statistics *are* the
                          parameters. Good general-purpose fallback drafter.
  * `PromptLookupDraft` -- zero-parameter: search the tokens generated so far
                          for the most recent earlier occurrence of the last
                          `lookup_n` tokens, and propose whatever followed
                          them last time. Extremely effective whenever the
                          model is likely to repeat itself against its own
                          context (summarization, code editing, RAG,
                          structured extraction) -- exactly HF transformers'
                          "prompt lookup decoding" / llama.cpp's `--draft-lookup`.

Both implement the same `propose(history, k) -> (tokens, probs)` interface
consumed by `engine.speculative.LightweightSpeculativeDecoder`.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Tuple


class NGramDraft:
    """Order-`n` statistical language model draft.

    `train()` is O(total tokens) and needs no gradient computation: it just
    counts `(context) -> Counter(next_token)`. `propose()` then greedily
    walks the most likely continuation up to `k` tokens, stopping early the
    first time it hits an unseen context (rather than guessing).
    """

    def __init__(self, n: int = 3):
        if n < 2:
            raise ValueError("n must be >= 2 (need at least 1 token of context)")
        self.n = n
        self.counts: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
        self.total_contexts_seen = 0

    def train(self, token_sequences: List[List[int]]):
        """token_sequences: a corpus of already-tokenized sequences (ids)."""
        ctx_len = self.n - 1
        for seq in token_sequences:
            for i in range(len(seq) - ctx_len):
                ctx = tuple(seq[i:i + ctx_len])
                nxt = seq[i + ctx_len]
                self.counts[ctx][nxt] += 1
                self.total_contexts_seen += 1

    def propose(self, history: List[int], k: int) -> Tuple[List[int], List[float]]:
        ctx_len = self.n - 1
        ctx = list(history[-ctx_len:]) if ctx_len > 0 else []
        tokens: List[int] = []
        probs: List[float] = []
        for _ in range(k):
            key = tuple(ctx[-ctx_len:])
            counter = self.counts.get(key)
            if not counter:
                break  # no statistics for this context -- stop rather than guess
            tok, cnt = counter.most_common(1)[0]
            total = sum(counter.values())
            tokens.append(tok)
            probs.append(cnt / total)
            ctx.append(tok)
        return tokens, probs

    @property
    def n_contexts(self) -> int:
        return len(self.counts)


class PromptLookupDraft:
    """Zero-parameter drafter: propose whatever token sequence followed the
    most recent earlier occurrence of the last `lookup_n` tokens in the
    context seen so far. Deterministic (assigns probability 1.0 to its own
    proposals, since it isn't sampling from any distribution)."""

    def __init__(self, lookup_n: int = 3, max_k: int = 8):
        self.lookup_n = lookup_n
        self.max_k = max_k

    def propose(self, history: List[int], k: int) -> Tuple[List[int], List[float]]:
        n = self.lookup_n
        if len(history) < n + 1:
            return [], []
        needle = history[-n:]
        limit = min(k, self.max_k)
        # Search backwards (excluding the trailing occurrence itself) for the
        # most recent earlier match, so the "predicted" continuation is the
        # most contextually-relevant one available.
        for start in range(len(history) - n - 1, -1, -1):
            if history[start:start + n] == needle:
                cont = history[start + n:start + n + limit]
                if cont:
                    return list(cont), [1.0] * len(cont)
        return [], []
