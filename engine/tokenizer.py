"""
A small, dependency-free byte-level BPE tokenizer.

Design goals:
  * No external deps (no `tokenizers`/`sentencepiece`) so the engine has zero
    third-party requirements for deployment, per the README.
  * Byte-level base vocabulary (256 tokens) guarantees lossless round-trip
    encode/decode for *any* input, even before any merges are learned.
  * `train()` learns merges from a text corpus using the standard BPE
    algorithm (most frequent adjacent pair -> new symbol, repeat).
  * Trained vocab/merges can be saved/loaded as JSON for reuse.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Dict, List, Tuple

_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


def _byte_to_unicode() -> Dict[int, str]:
    """Reversible byte -> printable-unicode mapping (GPT-2 style)."""
    bs = list(range(ord("!"), ord("~") + 1)) + \
        list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class BPETokenizer:
    """Byte-level BPE tokenizer with learnable merges."""

    def __init__(self):
        self._byte_encoder = _byte_to_unicode()
        self._byte_decoder = {v: k for k, v in self._byte_encoder.items()}

        # Base vocabulary: specials + all 256 byte-symbols.
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        for tok in _SPECIAL_TOKENS:
            self._add_token(tok)
        for b in range(256):
            self._add_token(self._byte_encoder[b])

        self.merges: List[Tuple[str, str]] = []
        self._merge_rank: Dict[Tuple[str, str], int] = {}

    # ------------------------------------------------------------------ #
    # vocabulary helpers
    # ------------------------------------------------------------------ #
    def _add_token(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        idx = len(self.token_to_id)
        self.token_to_id[tok] = idx
        self.id_to_token[idx] = tok
        return idx

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ------------------------------------------------------------------ #
    # training
    # ------------------------------------------------------------------ #
    def train(self, corpus: List[str] | str, num_merges: int = 1000, verbose: bool = False):
        """Learn `num_merges` BPE merge rules from a text corpus."""
        if isinstance(corpus, str):
            corpus = [corpus]

        # Represent each word as a tuple of byte-symbols.
        word_freq: Counter = Counter()
        for line in corpus:
            for word in line.split(" "):
                if word == "":
                    continue
                symbols = tuple(self._byte_encoder[b] for b in word.encode("utf-8"))
                word_freq[symbols] += 1

        for i in range(num_merges):
            pair_counts: Counter = Counter()
            for symbols, freq in word_freq.items():
                for a, b in zip(symbols, symbols[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            best_pair, best_count = pair_counts.most_common(1)[0]
            if best_count < 2:
                break

            new_symbol = best_pair[0] + best_pair[1]
            self._add_token(new_symbol)
            self.merges.append(best_pair)
            self._merge_rank[best_pair] = len(self.merges) - 1

            new_word_freq: Counter = Counter()
            for symbols, freq in word_freq.items():
                new_word_freq[self._merge_word(symbols, best_pair, new_symbol)] += freq
            word_freq = new_word_freq

            if verbose and (i + 1) % 100 == 0:
                print(f"[BPE] merge {i + 1}/{num_merges}: {best_pair} -> {new_symbol!r}")

    @staticmethod
    def _merge_word(symbols: Tuple[str, ...], pair: Tuple[str, str], new_symbol: str) -> Tuple[str, ...]:
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
                out.append(new_symbol)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        return tuple(out)

    # ------------------------------------------------------------------ #
    # encode / decode
    # ------------------------------------------------------------------ #
    def _bpe_merge(self, symbols: Tuple[str, ...]) -> Tuple[str, ...]:
        """Greedily apply learned merges (lowest rank first) to a symbol
        sequence. Works on any sequence, not just single "words" -- merges
        that were only ever observed within a word (e.g. never spanning a
        space) simply never match here, so word boundaries are respected
        without needing to special-case them at encode time."""
        if not self._merge_rank:
            return symbols
        symbols = list(symbols)
        while len(symbols) > 1:
            pairs = list(zip(symbols, symbols[1:]))
            ranked = [(self._merge_rank[p], p) for p in pairs if p in self._merge_rank]
            if not ranked:
                break
            _, best_pair = min(ranked, key=lambda x: x[0])
            symbols = list(self._merge_word(tuple(symbols), best_pair, best_pair[0] + best_pair[1]))
        return tuple(symbols)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.token_to_id["<bos>"])
        if text:
            # Encode the *entire* string (including spaces/punctuation) as one
            # byte-symbol sequence so decode() can reconstruct it exactly.
            symbols = tuple(self._byte_encoder[b] for b in text.encode("utf-8"))
            for sym in self._bpe_merge(symbols):
                ids.append(self.token_to_id.get(sym, self.token_to_id["<unk>"]))
        if add_eos:
            ids.append(self.token_to_id["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        chars: List[str] = []
        for i in ids:
            tok = self.id_to_token.get(i, "")
            if tok in _SPECIAL_TOKENS:
                continue
            chars.append(tok)
        text = "".join(chars)
        raw = bytes(self._byte_decoder[c] for c in text if c in self._byte_decoder)
        return raw.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "merges": self.merges,
            }, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok.token_to_id = {k: int(v) for k, v in data["token_to_id"].items()}
        tok.id_to_token = {v: k for k, v in tok.token_to_id.items()}
        tok.merges = [tuple(p) for p in data["merges"]]
        tok._merge_rank = {p: i for i, p in enumerate(tok.merges)}
        return tok
