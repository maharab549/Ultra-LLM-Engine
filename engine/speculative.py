"""
Speculative decoding (Leviathan et al. 2023 / Chen et al. 2023).

A small, cheap "draft" model proposes K tokens autoregressively. The big
"target" model then evaluates all K draft positions and, for each one,
accepts the draft token with probability min(1, p_target / q_draft) --
exactly reproducing the target model's own sampling distribution while only
paying the offload/compute cost of the big model once per K tokens instead
of once per token. On the first rejection, a corrective token is resampled
from the residual distribution max(0, p_target - q_draft) and everything
after it is discarded; if all K are accepted, one bonus token is drawn from
the target's next-step distribution "for free".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .model import KVCache, QuantizedTransformer


@dataclass
class SpecStepResult:
    tokens: List[int]              # accepted (and possibly one bonus/corrective) tokens
    n_accepted: int                # how many of the K draft tokens were accepted as-is
    n_proposed: int                # K
    cache_valid_length: int         # number of *newly appended* cache entries
                                     # (this round only) that are valid. The
                                     # caller must truncate each cache to
                                     # (length_before_this_round + cache_valid_length),
                                     # NOT to cache_valid_length directly --
                                     # see Scheduler.generate().


class SpeculativeDecoder:
    def __init__(self, draft_model: QuantizedTransformer, target_model: QuantizedTransformer,
                 k: int = 5, draft_temperature: float = 1.0, target_temperature: float = 1.0,
                 seed: int = 0, adaptive: bool = False, min_k: int = 1, max_k: int = 16):
        self.draft = draft_model
        self.target = target_model
        self.k = k
        self.draft_temp = draft_temperature
        self.target_temp = target_temperature
        self.rng = np.random.default_rng(seed)

        # Adaptive-K: rounds where the draft is being accepted almost every
        # time are "leaving throughput on the table" -- propose more next
        # round. Rounds with poor acceptance waste target-model compute on
        # draft tokens that get thrown away -- propose fewer. A simple
        # exponential moving average of the per-round acceptance rate drives
        # a bounded random walk in `k`, so it adapts smoothly instead of
        # oscillating on every single round.
        self.adaptive = adaptive
        self.min_k = min_k
        self.max_k = max_k
        self._ema_accept_rate: Optional[float] = None
        self._ema_alpha = 0.3
        self.stats = {"rounds": 0, "total_accepted": 0, "total_proposed": 0}

    def generate_step(self, last_token: int, pos: int,
                       draft_cache: KVCache, target_cache: KVCache) -> SpecStepResult:
        """One round of speculative decoding starting from `last_token` at position `pos`.

        Returns the tokens that should be appended to the sequence. Caller is
        responsible for advancing `pos` by `len(result.tokens)` and, if a
        rejection happened, truncating both caches back to the accepted
        length before the next round (see `Scheduler` for the bookkeeping).
        """
        # ---- 1. draft model proposes K tokens -------------------------------
        draft_tokens: List[int] = []
        draft_probs: List[np.ndarray] = []
        tok = last_token
        p = pos
        for _ in range(self.k):
            logits = self.draft.forward_token(tok, p, draft_cache)
            tok, probs = self.draft.sample(logits, self.draft_temp, self.rng)
            draft_tokens.append(tok)
            draft_probs.append(probs)
            p += 1

        # ---- 2. target model verifies each draft position --------------------
        # (In a fully batched implementation this is a single forward pass over
        # all K positions; here we replay them sequentially through the target
        # model's own KV cache, which is functionally equivalent for a
        # reference/toy engine and keeps the attention implementation simple.)
        target_probs: List[np.ndarray] = []
        tok = last_token
        p = pos
        for i in range(self.k):
            logits = self.target.forward_token(tok, p, target_cache)
            probs = self._softmax(logits / max(self.target_temp, 1e-6))
            target_probs.append(probs)
            tok = draft_tokens[i]
            p += 1

        # ---- 3. accept/reject each draft token --------------------------------
        accepted: List[int] = []
        n_accepted = 0
        for i in range(self.k):
            x_i = draft_tokens[i]
            p_target = target_probs[i][x_i]
            q_draft = draft_probs[i][x_i]
            accept_prob = min(1.0, float(p_target) / float(q_draft + 1e-12))
            if self.rng.random() < accept_prob:
                accepted.append(x_i)
                n_accepted += 1
            else:
                # Reject: resample from the residual distribution and stop.
                residual = np.clip(target_probs[i] - draft_probs[i], 0, None)
                total = residual.sum()
                if total <= 1e-12:
                    residual = target_probs[i]
                else:
                    residual = residual / total
                corrective = int(self.rng.choice(len(residual), p=residual))
                accepted.append(corrective)
                # Valid cache entries cover positions pos..pos+i (i.e. last_token
                # plus the i accepted draft tokens) = i + 1 entries.
                self._record_round(n_accepted, self.k)
                return SpecStepResult(tokens=accepted, n_accepted=n_accepted,
                                       n_proposed=self.k, cache_valid_length=n_accepted + 1)

        # All K draft tokens accepted -> draw one bonus token for free from the
        # target model's distribution at the final verified position.
        bonus_logits = self.target.forward_token(accepted[-1], pos + self.k, target_cache)
        bonus_tok, _ = self.target.sample(bonus_logits, self.target_temp, self.rng)
        accepted.append(bonus_tok)
        # The draft model never ran on its own last proposed token as *input*
        # (there was no need to, since we don't ask it to predict past K).
        # Run that one extra cache-fill step so draft_cache stays position-
        # aligned with target_cache for the next round.
        self.draft.forward_token(draft_tokens[-1], pos + self.k, draft_cache)
        self._record_round(n_accepted, self.k)
        return SpecStepResult(tokens=accepted, n_accepted=n_accepted,
                               n_proposed=self.k, cache_valid_length=self.k + 1)

    def _record_round(self, n_accepted: int, n_proposed: int):
        self.stats["rounds"] += 1
        self.stats["total_accepted"] += n_accepted
        self.stats["total_proposed"] += n_proposed
        if not self.adaptive or n_proposed == 0:
            return
        rate = n_accepted / n_proposed
        self._ema_accept_rate = (rate if self._ema_accept_rate is None
                                  else self._ema_alpha * rate + (1 - self._ema_alpha) * self._ema_accept_rate)
        if self._ema_accept_rate > 0.9 and self.k < self.max_k:
            self.k += 1
        elif self._ema_accept_rate < 0.5 and self.k > self.min_k:
            self.k -= 1

    @property
    def acceptance_rate(self) -> float:
        total = self.stats["total_proposed"]
        return self.stats["total_accepted"] / total if total else 0.0

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()


class LightweightSpeculativeDecoder:
    """Speculative decoding using a statistical drafter (`NGramDraft` /
    `PromptLookupDraft` from `engine.draft_strategies`) instead of a second
    neural network. No draft KV cache is needed at all -- the drafter reads
    directly off the token history -- so bookkeeping only touches
    `target_cache`.

    Verification has two modes, chosen automatically from `target_temperature`:

      * **Greedy target** (`target_temperature <= 1e-6`): a draft token is
        accepted iff it equals the target model's own argmax at that
        position. This exactly reproduces greedy target decoding -- not an
        approximation -- because a greedy target's output is a deterministic
        function of the prefix, and speculative decoding is just checking
        that function without paying to run it token-by-token.
      * **Sampling target** (`target_temperature > 0`): true speculative
        rejection sampling needs `q(x)` (draft probability) over the *whole*
        vocabulary to build the residual distribution on rejection.
        Statistical drafters only produce a probability for their own
        proposed token, not a full distribution, so on rejection we fall
        back to resampling directly from the target's own distribution
        `p_target` (skipping the residual subtraction). This is a documented
        approximation: accepted tokens are still exactly distributed
        according to `min(1, p_target(x)/q(x))`, but the corrective token on
        rejection is drawn from `p_target` rather than the theoretically
        exact residual `max(0, p_target - q)`.
    """

    def __init__(self, drafter, target_model: QuantizedTransformer, k: int = 5,
                 target_temperature: float = 1.0, seed: int = 0):
        self.drafter = drafter
        self.target = target_model
        self.k = k
        self.target_temp = target_temperature
        self.rng = np.random.default_rng(seed)
        self.stats = {"rounds": 0, "total_accepted": 0, "total_proposed": 0}

    def generate_step(self, history: List[int], pos: int, target_cache: KVCache) -> SpecStepResult:
        draft_tokens, draft_probs = self.drafter.propose(history, self.k)
        n_proposed = len(draft_tokens)
        greedy = self.target_temp <= 1e-6

        if n_proposed == 0:
            # Drafter had nothing to propose (e.g. no matching n-gram context
            # yet) -- fall back to a single ordinary target-model decode step.
            logits = self.target.forward_token(history[-1], pos, target_cache)
            tok, _ = self.target.sample(logits, self.target_temp, self.rng)
            self._record_round(0, 0)
            return SpecStepResult(tokens=[tok], n_accepted=0, n_proposed=0, cache_valid_length=1)

        tok = history[-1]
        p = pos
        target_probs: List[np.ndarray] = []
        for i in range(n_proposed):
            logits = self.target.forward_token(tok, p, target_cache)
            probs = self._softmax(logits / max(self.target_temp, 1e-6))
            target_probs.append(probs)
            tok = draft_tokens[i]
            p += 1

        accepted: List[int] = []
        n_accepted = 0
        for i in range(n_proposed):
            x_i = draft_tokens[i]
            if greedy:
                target_argmax = int(np.argmax(target_probs[i]))
                accept = (x_i == target_argmax)
            else:
                q_draft = draft_probs[i]
                accept_prob = min(1.0, float(target_probs[i][x_i]) / float(q_draft + 1e-12))
                accept = self.rng.random() < accept_prob

            if accept:
                accepted.append(x_i)
                n_accepted += 1
                continue

            corrective = (int(np.argmax(target_probs[i])) if greedy
                          else int(self.rng.choice(len(target_probs[i]), p=target_probs[i])))
            accepted.append(corrective)
            self._record_round(n_accepted, n_proposed)
            return SpecStepResult(tokens=accepted, n_accepted=n_accepted,
                                   n_proposed=n_proposed, cache_valid_length=n_accepted + 1)

        # All proposed tokens accepted -> bonus token, same as the neural-draft path.
        bonus_logits = self.target.forward_token(accepted[-1], pos + n_proposed, target_cache)
        bonus_tok, _ = self.target.sample(bonus_logits, self.target_temp, self.rng)
        accepted.append(bonus_tok)
        self._record_round(n_accepted, n_proposed)
        return SpecStepResult(tokens=accepted, n_accepted=n_accepted,
                               n_proposed=n_proposed, cache_valid_length=n_proposed + 1)

    def _record_round(self, n_accepted: int, n_proposed: int):
        self.stats["rounds"] += 1
        self.stats["total_accepted"] += n_accepted
        self.stats["total_proposed"] += n_proposed

    @property
    def acceptance_rate(self) -> float:
        total = self.stats["total_proposed"]
        return self.stats["total_accepted"] / total if total else 0.0

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()
