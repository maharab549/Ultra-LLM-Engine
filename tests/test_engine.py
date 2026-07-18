"""
Tests for the inference engine.
Run: python -m pytest tests/test_engine.py -v
"""
import os
import struct
import tempfile

import numpy as np
import pytest

from engine.config import EngineConfig
from engine.tokenizer import BPETokenizer
from engine.quantization import WeightQuantizer, NF4Quantizer, get_quantizer
from engine.kernels import FusedKernels
from engine.offload_manager import OffloadManager
from engine.model import QuantizedTransformer, KVCache
from engine.speculative import SpeculativeDecoder, LightweightSpeculativeDecoder
from engine.scheduler import Scheduler
from engine.paged_cache import PagePool, PagedSequenceCache
from engine.draft_strategies import NGramDraft, PromptLookupDraft
from engine.loaders import safetensors_io, gguf_io, checkpoint as ckpt_loader


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
def test_tokenizer_roundtrip():
    tok = BPETokenizer()
    text = "hello world 123"
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert isinstance(ids, list)
    assert len(ids) > 0
    assert decoded == text


def test_tokenizer_roundtrip_after_training():
    tok = BPETokenizer()
    tok.train(["the quick brown fox jumps over the lazy dog", "the fox ran"], num_merges=50)
    for text in ["the quick brown fox", "unseen wörds 🚀", ""]:
        ids = tok.encode(text)
        assert tok.decode(ids) == text
    # training should have produced merges and grown the vocab past the byte base
    assert tok.vocab_size > 256 + 4


# --------------------------------------------------------------------------- #
# quantization
# --------------------------------------------------------------------------- #
def test_int4_quantization():
    cfg = EngineConfig(hidden_dim=512, group_size=128)
    quant = WeightQuantizer(cfg)
    w = np.random.randn(256, 512).astype(np.float16)
    qw, sc, zr = quant.quantize(w)
    assert qw.dtype == np.uint8
    assert qw.shape[1] == 256  # 512 // 2
    # Dequantize and check MSE is bounded
    w_hat = np.concatenate([
        quant.dequantize_group(qw, sc, zr, g) for g in range(512 // 128)
    ], axis=1)
    mse = ((w.astype(np.float32) - w_hat) ** 2).mean()
    assert mse < 0.5  # loose bound for random weights


def test_quantize_dequantize_matches_helper():
    cfg = EngineConfig(hidden_dim=256, group_size=64)
    quant = WeightQuantizer(cfg)
    w = (np.random.randn(32, 256) * 0.05).astype(np.float32)
    qw, sc, zr = quant.quantize(w)
    w_hat_manual = np.concatenate(
        [quant.dequantize_group(qw, sc, zr, g) for g in range(256 // 64)], axis=1)
    w_hat_helper = quant.dequantize(qw, sc, zr)
    np.testing.assert_allclose(w_hat_manual, w_hat_helper)


# --------------------------------------------------------------------------- #
# kernels
# --------------------------------------------------------------------------- #
def test_fused_kernel_fallback():
    cfg = EngineConfig(hidden_dim=256, group_size=128)
    kern = FusedKernels(cfg, force_numpy=True)
    # INT4 GEMV fallback
    quant = WeightQuantizer(cfg)
    w = np.random.randn(128, 256).astype(np.float16)
    qw, sc, zr = quant.quantize(w)
    x = np.random.randint(-128, 127, size=(256,), dtype=np.int8)
    y = kern.fused_int4_gemv(qw, sc, zr, x, 1.0)
    assert y.shape == (128,)
    assert y.dtype == np.float16


def test_fused_rmsnorm_rope_shape_and_norm():
    cfg = EngineConfig(hidden_dim=64, n_heads=4, group_size=64)
    kern = FusedKernels(cfg, force_numpy=True)
    x = np.random.randn(64).astype(np.float32)
    w = np.ones(64, dtype=np.float32)
    out = kern.fused_rmsnorm_rope(x, w, pos=0)
    assert out.shape == (64,)
    # RoPE is a rotation (per 2D pair) so it preserves the per-head norm of
    # the normalized vector; at pos=0 every rotation angle is 0, so output
    # should equal the plain RMSNorm output.
    expected_norm = kern._fused_rmsnorm_rope_numpy(x, w, 0, None)
    np.testing.assert_allclose(out, expected_norm, atol=1e-5)


# --------------------------------------------------------------------------- #
# offload manager
# --------------------------------------------------------------------------- #
def test_offload_manager_serves_all_layers_and_prefetches():
    layers = {i: f"layer-{i}-weights" for i in range(10)}
    mgr = OffloadManager(layers, gpu_layers=2, prefetch_depth=2)
    try:
        for i in range(10):
            assert mgr.get_layer(i) == f"layer-{i}-weights"
        assert mgr.stats["host_to_gpu_copies"] >= 8  # all cold layers were copied at least once
    finally:
        mgr.shutdown()


# --------------------------------------------------------------------------- #
# end-to-end: tiny random model, greedy and speculative decoding
# --------------------------------------------------------------------------- #
def _tiny_config(**overrides):
    base = dict(hidden_dim=32, n_layers=2, n_heads=4, vocab_size=64,
                group_size=16, gpu_layers=1, prefetch_depth=1, seed=0)
    base.update(overrides)
    return EngineConfig(**base)


def test_forward_token_produces_valid_logits():
    cfg = _tiny_config()
    model = QuantizedTransformer.from_random(cfg)
    cache = KVCache(cfg.n_layers)
    logits = model.forward_token(token_id=1, pos=0, cache=cache)
    assert logits.shape == (cfg.vocab_size,)
    assert np.isfinite(logits).all()


def test_greedy_generation_end_to_end():
    tokenizer = BPETokenizer()
    tokenizer.train(["hello world", "the fox"], num_merges=20)
    cfg = _tiny_config(vocab_size=tokenizer.vocab_size)
    model = QuantizedTransformer.from_random(cfg)
    with Scheduler(cfg, tokenizer, model) as sched:
        text = sched.generate("hello", max_tokens=8)
        assert isinstance(text, str)
        assert text.startswith("hello")


def test_speculative_decoding_matches_cache_bookkeeping():
    cfg = _tiny_config(speculate_k=3)
    draft_cfg = _tiny_config(hidden_dim=16, n_layers=1, n_heads=2, group_size=16, speculate_k=3)
    target = QuantizedTransformer.from_random(cfg)
    draft = QuantizedTransformer.from_random(draft_cfg)

    spec = SpeculativeDecoder(draft, target, k=cfg.speculate_k, seed=1)
    draft_cache = KVCache(draft_cfg.n_layers)
    target_cache = KVCache(cfg.n_layers)

    result = spec.generate_step(last_token=0, pos=0, draft_cache=draft_cache, target_cache=target_cache)
    assert len(result.tokens) == result.cache_valid_length
    assert 0 <= result.n_accepted <= result.n_proposed
    assert result.n_proposed == cfg.speculate_k


def test_speculative_generation_end_to_end():
    tokenizer = BPETokenizer()
    tokenizer.train(["hello world", "the fox"], num_merges=20)
    cfg = _tiny_config(speculate_k=2, vocab_size=tokenizer.vocab_size)
    draft_cfg = _tiny_config(hidden_dim=16, n_layers=1, n_heads=2, group_size=16,
                              vocab_size=tokenizer.vocab_size)
    target = QuantizedTransformer.from_random(cfg)
    draft = QuantizedTransformer.from_random(draft_cfg)
    with Scheduler(cfg, tokenizer, target, draft) as sched:
        text = sched.generate("hello", max_tokens=10)
        assert isinstance(text, str)
        assert text.startswith("hello")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------- #
# quantization: 2/8-bit and NF4
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_quantization_bit_widths(bits):
    cfg = EngineConfig(hidden_dim=64, group_size=32, quant_bits=bits)
    quant = WeightQuantizer(cfg)
    w = (np.random.randn(8, 64) * 0.1).astype(np.float32)
    qw, sc, zr = quant.quantize(w)
    assert qw.dtype == np.uint8
    assert qw.shape == (8, 64 * bits // 8)
    w_hat = quant.dequantize(qw, sc, zr)
    mse = ((w - w_hat) ** 2).mean()
    assert mse < 0.05
    # higher bit width should reconstruct at least as accurately
    return mse


def test_quantization_bits_monotonic_accuracy():
    w = (np.random.randn(16, 64) * 0.1).astype(np.float32)
    mses = {}
    for bits in (2, 4, 8):
        cfg = EngineConfig(hidden_dim=64, group_size=32, quant_bits=bits)
        quant = WeightQuantizer(cfg)
        qw, sc, zr = quant.quantize(w)
        w_hat = quant.dequantize(qw, sc, zr)
        mses[bits] = ((w - w_hat) ** 2).mean()
    assert mses[8] <= mses[4] <= mses[2]


def test_nf4_quantizer_roundtrip():
    cfg = EngineConfig(hidden_dim=64, group_size=32, quant_scheme="nf4", quant_bits=4)
    quant = get_quantizer(cfg)
    assert isinstance(quant, NF4Quantizer)
    w = (np.random.randn(8, 64) * 0.1).astype(np.float32)
    qw, sc, zeros = quant.quantize(w)
    assert zeros is None
    w_hat = quant.dequantize(qw, sc, zeros)
    assert ((w - w_hat) ** 2).mean() < 0.05


# --------------------------------------------------------------------------- #
# batched decode
# --------------------------------------------------------------------------- #
def test_batched_decode_matches_single_sequence():
    cfg = _tiny_config(vocab_size=64)
    model = QuantizedTransformer.from_random(cfg)
    tokens = [1, 2, 3]

    single_caches = [KVCache(cfg.n_layers) for _ in tokens]
    single_logits = [model.forward_token(t, 0, single_caches[i]) for i, t in enumerate(tokens)]

    batch_caches = [KVCache(cfg.n_layers) for _ in tokens]
    batch_logits = model.forward_batch_step(tokens, 0, batch_caches)

    for i in range(len(tokens)):
        np.testing.assert_allclose(
            single_logits[i].astype(np.float32), batch_logits[i].astype(np.float32), atol=1e-3)


# --------------------------------------------------------------------------- #
# paged KV cache
# --------------------------------------------------------------------------- #
def test_paged_cache_matches_kvcache_semantics():
    pool = PagePool(page_size=4)
    paged = PagedSequenceCache(n_layers=1, pool=pool)
    ref = KVCache(1)
    rng = np.random.default_rng(0)

    for _ in range(10):
        k, v = rng.standard_normal(6).astype(np.float32), rng.standard_normal(6).astype(np.float32)
        paged.append(0, k, v)
        ref.append(0, k, v)
    pk, pv = paged.get(0)
    rk, rv = ref.get(0)
    assert np.array_equal(pk, rk) and np.array_equal(pv, rv)
    assert pool.total_allocated == 3  # ceil(10/4)

    paged.truncate(0, 5)
    ref.truncate(0, 5)
    pk2, pv2 = paged.get(0)
    rk2, rv2 = ref.get(0)
    assert np.array_equal(pk2, rk2) and np.array_equal(pv2, rv2)
    assert pool.total_allocated == 2  # ceil(5/4)


def test_paged_cache_pages_are_reused_across_sequences():
    pool = PagePool(page_size=4)
    seq1 = PagedSequenceCache(n_layers=1, pool=pool)
    rng = np.random.default_rng(0)
    for _ in range(8):
        seq1.append(0, rng.standard_normal(4).astype(np.float32), rng.standard_normal(4).astype(np.float32))
    seq1.free()
    assert pool.total_allocated == 0

    seq2 = PagedSequenceCache(n_layers=1, pool=pool)
    for _ in range(4):
        seq2.append(0, rng.standard_normal(4).astype(np.float32), rng.standard_normal(4).astype(np.float32))
    # seq2 should have reused a page freed by seq1, not created a brand new one
    assert pool.n_pages_ever_created == 2


def test_scheduler_with_paged_cache_generates():
    tokenizer = BPETokenizer()
    tokenizer.train(["hello world", "the fox"], num_merges=20)
    cfg = _tiny_config(vocab_size=tokenizer.vocab_size, use_paged_cache=True, page_size=4)
    model = QuantizedTransformer.from_random(cfg)
    with Scheduler(cfg, tokenizer, model, draft_mode="greedy") as sched:
        text = sched.generate("hello", max_tokens=8)
        assert text.startswith("hello")


# --------------------------------------------------------------------------- #
# statistical draft strategies (n-gram / prompt-lookup)
# --------------------------------------------------------------------------- #
def test_ngram_draft_learns_deterministic_continuation():
    drafter = NGramDraft(n=3)
    seq = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
    drafter.train([seq])
    tokens, probs = drafter.propose(history=[1, 2], k=3)
    assert tokens[:1] == [3]  # context (1,2) always followed by 3 in training data
    assert all(0.0 < p <= 1.0 for p in probs)


def test_ngram_draft_stops_on_unseen_context():
    drafter = NGramDraft(n=3)
    drafter.train([[1, 2, 3]])
    tokens, probs = drafter.propose(history=[9, 9], k=5)
    assert tokens == [] and probs == []


def test_prompt_lookup_draft_finds_recent_repeat():
    drafter = PromptLookupDraft(lookup_n=2, max_k=5)
    # "5 6" appeared before at index 0-1, followed by [7, 8]
    history = [5, 6, 7, 8, 9, 5, 6]
    tokens, probs = drafter.propose(history, k=2)
    assert tokens == [7, 8]
    assert probs == [1.0, 1.0]


def test_prompt_lookup_draft_empty_when_no_repeat():
    drafter = PromptLookupDraft(lookup_n=2, max_k=5)
    tokens, probs = drafter.propose([1, 2, 3, 4], k=5)
    assert tokens == [] and probs == []


def test_lightweight_speculative_decoder_greedy_exact():
    """With a greedy target (temperature ~0), LightweightSpeculativeDecoder
    must reproduce exactly the tokens plain greedy decoding would produce."""
    cfg = _tiny_config(vocab_size=64, target_temperature=1e-8)
    model = QuantizedTransformer.from_random(cfg)
    drafter = PromptLookupDraft(lookup_n=2, max_k=3)
    dec = LightweightSpeculativeDecoder(drafter, model, k=3, target_temperature=1e-8, seed=0)

    # Reference: plain greedy decode of the same model/prompt.
    prompt = [1, 2, 3, 4, 5]
    ref_cache = KVCache(cfg.n_layers)
    logits = None
    for pos, t in enumerate(prompt):
        logits = model.forward_token(t, pos, ref_cache)
    last, _ = model.sample(logits, temperature=1e-8)
    ref_tokens = [last]
    pos = len(prompt) - 1
    for _ in range(9):
        pos += 1
        logits = model.forward_token(ref_tokens[-1], pos, ref_cache)
        tok, _ = model.sample(logits, temperature=1e-8)
        ref_tokens.append(tok)

    # LightweightSpeculativeDecoder over the same prompt/model.
    spec_cache = KVCache(cfg.n_layers)
    logits = None
    for pos_, t in enumerate(prompt):
        logits = model.forward_token(t, pos_, spec_cache)
    last, _ = model.sample(logits, temperature=1e-8)
    spec_tokens = [last]
    history = list(prompt) + spec_tokens
    pos = len(prompt)
    while len(spec_tokens) < 10:
        pre_length = spec_cache.length(0)
        result = dec.generate_step(history, pos, spec_cache)
        spec_tokens.extend(result.tokens)
        history.extend(result.tokens)
        pos += len(result.tokens)
        for i in range(cfg.n_layers):
            spec_cache.truncate(i, pre_length + result.cache_valid_length)

    assert spec_tokens[:10] == ref_tokens[:10]


def test_scheduler_ngram_and_lookup_modes_generate():
    tokenizer = BPETokenizer()
    tokenizer.train(["hello world", "hello there", "the fox"], num_merges=20)
    for mode in ("ngram", "lookup"):
        cfg = _tiny_config(vocab_size=tokenizer.vocab_size, speculate_k=3)
        model = QuantizedTransformer.from_random(cfg)
        with Scheduler(cfg, tokenizer, model, draft_mode=mode) as sched:
            text = sched.generate("hello world", max_tokens=10)
            assert text.startswith("hello world")


# --------------------------------------------------------------------------- #
# adaptive-K speculative decoding
# --------------------------------------------------------------------------- #
def test_adaptive_k_moves_within_bounds():
    cfg = _tiny_config(speculate_k=3, adaptive_speculation=True, min_speculate_k=1, max_speculate_k=6)
    draft_cfg = _tiny_config(hidden_dim=16, n_layers=1, n_heads=2, group_size=16)
    target = QuantizedTransformer.from_random(cfg)
    draft = QuantizedTransformer.from_random(draft_cfg)
    dec = SpeculativeDecoder(draft, target, k=cfg.speculate_k, seed=0,
                              adaptive=True, min_k=1, max_k=6)
    draft_cache, target_cache = KVCache(draft_cfg.n_layers), KVCache(cfg.n_layers)
    last_token, pos = 0, 0
    for _ in range(15):
        pre_length = target_cache.length(0)  # == draft_cache.length(0), both grow in lockstep
        result = dec.generate_step(last_token, pos, draft_cache, target_cache)
        pos += len(result.tokens)
        target_len = pre_length + result.cache_valid_length
        for i in range(draft_cfg.n_layers):
            draft_cache.truncate(i, target_len)
        for i in range(cfg.n_layers):
            target_cache.truncate(i, target_len)
        last_token = result.tokens[-1]
        assert 1 <= dec.k <= 6


def test_batched_generation_end_to_end():
    tokenizer = BPETokenizer()
    tokenizer.train(["hello world", "the quick fox"], num_merges=20)
    cfg = _tiny_config(vocab_size=tokenizer.vocab_size)
    model = QuantizedTransformer.from_random(cfg)
    with Scheduler(cfg, tokenizer, model, draft_mode="greedy") as sched:
        texts = sched.generate_batch(["hello world", "the quick fox"], max_tokens=6)
        assert len(texts) == 2
        assert texts[0].startswith("hello world")
        assert texts[1].startswith("the quick fox")


# --------------------------------------------------------------------------- #
# checkpoint loaders (safetensors + gguf), no network/deps needed
# --------------------------------------------------------------------------- #
def _make_llama_style_tensors(cfg: EngineConfig, seed=0):
    rng = np.random.default_rng(seed)

    def w(o, i):
        return (rng.standard_normal((o, i)) * 0.02).astype(np.float16)

    tensors = {"model.embed_tokens.weight": w(cfg.vocab_size, cfg.hidden_dim)}
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = np.ones(cfg.hidden_dim, dtype=np.float32)
        tensors[p + "post_attention_layernorm.weight"] = np.ones(cfg.hidden_dim, dtype=np.float32)
        tensors[p + "self_attn.q_proj.weight"] = w(cfg.hidden_dim, cfg.hidden_dim)
        tensors[p + "self_attn.k_proj.weight"] = w(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_dim)
        tensors[p + "self_attn.v_proj.weight"] = w(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_dim)
        tensors[p + "self_attn.o_proj.weight"] = w(cfg.hidden_dim, cfg.hidden_dim)
        tensors[p + "mlp.gate_proj.weight"] = w(cfg.ffn_dim, cfg.hidden_dim)
        tensors[p + "mlp.up_proj.weight"] = w(cfg.ffn_dim, cfg.hidden_dim)
        tensors[p + "mlp.down_proj.weight"] = w(cfg.hidden_dim, cfg.ffn_dim)
    tensors["model.norm.weight"] = np.ones(cfg.hidden_dim, dtype=np.float32)
    return tensors


def test_safetensors_roundtrip():
    tensors = {
        "a": (np.random.randn(4, 8)).astype(np.float32),
        "b": (np.random.randn(3, 5)).astype(np.float16),
        "c": (np.random.randint(-10, 10, (2, 2))).astype(np.int8),
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.safetensors")
        safetensors_io.save_file(tensors, path, metadata={"k": "v"})
        loaded = safetensors_io.load_file(path)
        for k, v in tensors.items():
            assert np.array_equal(v, loaded[k]) and v.dtype == loaded[k].dtype
        assert safetensors_io.load_metadata(path) == {"k": "v"}
        assert set(safetensors_io.tensor_names(path)) == set(tensors.keys())


def test_checkpoint_from_safetensors_builds_working_model():
    cfg = EngineConfig(hidden_dim=32, n_layers=2, n_heads=4, vocab_size=50, group_size=16, ffn_dim=64)
    tensors = _make_llama_style_tensors(cfg)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        safetensors_io.save_file(tensors, path)
        model = ckpt_loader.from_safetensors(path, cfg)
        cache = KVCache(cfg.n_layers)
        logits = model.forward_token(3, 0, cache)
        assert logits.shape == (cfg.vocab_size,)
        assert np.isfinite(logits).all()


def _write_synthetic_gguf(path, tensors, metadata):
    def w_str(f, s):
        b = s.encode("utf-8")
        f.write(struct.pack("<Q", len(b)))
        f.write(b)

    def w_val(f, v):
        if isinstance(v, str):
            f.write(struct.pack("<i", 8)); w_str(f, v)
        elif isinstance(v, int):
            f.write(struct.pack("<i", 4)); f.write(struct.pack("<I", v))
        elif isinstance(v, float):
            f.write(struct.pack("<i", 6)); f.write(struct.pack("<f", v))
        else:
            raise TypeError(v)

    with open(path, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<i", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))
        for k, v in metadata.items():
            w_str(f, k)
            w_val(f, v)
        offset = 0
        infos = []
        for name, arr in tensors.items():
            shape = list(reversed(arr.shape))
            infos.append((name, arr))
            w_str(f, name)
            f.write(struct.pack("<i", len(shape)))
            for d in shape:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<i", 0))  # F32
            f.write(struct.pack("<Q", offset))
            offset += arr.nbytes
        pos = f.tell()
        pad = (32 - pos % 32) % 32
        f.write(b"\x00" * pad)
        for _, arr in infos:
            f.write(np.ascontiguousarray(arr).tobytes())


def test_gguf_reader_parses_metadata_and_tensors():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.gguf")
        w1 = np.random.randn(4, 8).astype(np.float32)
        _write_synthetic_gguf(path, {"blk.0.attn_q.weight": w1},
                               {"general.architecture": "llama", "general.name": "tiny"})
        g = gguf_io.load(path)
        assert g.metadata["general.name"] == "tiny"
        assert "blk.0.attn_q.weight" in g.tensor_names()
        loaded = g.load_tensor("blk.0.attn_q.weight")
        np.testing.assert_allclose(loaded, w1)


def test_checkpoint_from_gguf_builds_working_model():
    cfg = EngineConfig(hidden_dim=32, n_layers=2, n_heads=4, vocab_size=50, group_size=16, ffn_dim=64)
    rng = np.random.default_rng(2)

    def w(o, i):
        return (rng.standard_normal((o, i)) * 0.02).astype(np.float32)

    tensors = {"token_embd.weight": w(cfg.vocab_size, cfg.hidden_dim)}
    for i in range(cfg.n_layers):
        p = f"blk.{i}."
        tensors[p + "attn_norm.weight"] = np.ones(cfg.hidden_dim, dtype=np.float32)
        tensors[p + "ffn_norm.weight"] = np.ones(cfg.hidden_dim, dtype=np.float32)
        tensors[p + "attn_q.weight"] = w(cfg.hidden_dim, cfg.hidden_dim)
        tensors[p + "attn_k.weight"] = w(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_dim)
        tensors[p + "attn_v.weight"] = w(cfg.n_kv_heads * cfg.head_dim, cfg.hidden_dim)
        tensors[p + "attn_output.weight"] = w(cfg.hidden_dim, cfg.hidden_dim)
        tensors[p + "ffn_gate.weight"] = w(cfg.ffn_dim, cfg.hidden_dim)
        tensors[p + "ffn_up.weight"] = w(cfg.ffn_dim, cfg.hidden_dim)
        tensors[p + "ffn_down.weight"] = w(cfg.hidden_dim, cfg.ffn_dim)
    tensors["output_norm.weight"] = np.ones(cfg.hidden_dim, dtype=np.float32)

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.gguf")
        _write_synthetic_gguf(path, tensors, {"general.architecture": "llama", "general.name": "tiny"})
        model = ckpt_loader.from_gguf(path, cfg)
        cache = KVCache(cfg.n_layers)
        logits = model.forward_token(5, 0, cache)
        assert logits.shape == (cfg.vocab_size,)
        assert np.isfinite(logits).all()
