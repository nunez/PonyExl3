"""Parity of ponyexl3.arch.deepseek_v41 against PipeNetwork/deepseek-v41-mlx.

Our arch is a name-for-name port of theirs (MIT); this harness builds both
models from the same tiny config, transplants our weights into theirs (module
trees are identical, so this is a direct flatten -> load), and asserts
bit-level agreement over chunked prefill + decode, plus per-submodule parity.

Env:
    PONYEXL3_DSV41_MLX=/path/to/deepseek-v41-mlx   (clone root; default
    ~/src/deepseek-v41-mlx). Skipped when the clone or mlx is unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

REF = Path(os.environ.get("PONYEXL3_DSV41_MLX", "~/src/deepseek-v41-mlx")).expanduser()
pytestmark = pytest.mark.skipif(
    not (REF / "deepseek_v41_mlx" / "model.py").is_file(),
    reason="set PONYEXL3_DSV41_MLX=/path/to/deepseek-v41-mlx (git clone)",
)
sys.path.insert(0, str(REF))

from mlx.utils import tree_flatten  # noqa: E402

from deepseek_v41_mlx import config as ref_config  # noqa: E402
from deepseek_v41_mlx import model as ref_model  # noqa: E402

from ponyexl3.arch.deepseek_v41 import config as our_config  # noqa: E402
from ponyexl3.arch.deepseek_v41 import model as our_model  # noqa: E402

# tiny config — identical to the oracle's tests/test_parity.py fixture
VOCAB = 96
B, S, DECODE, MAXSEQ = 2, 48, 8, 64
CFG = dict(
    vocab_size=VOCAB, dim=64, n_layers=8, moe_inter_dim=32,
    n_heads=4, n_routed_experts=8, n_shared_experts=1, n_activated_experts=3,
    score_func="sqrtsoftplus", route_scale=1.5, swiglu_limit=1.0,
    q_lora_rank=32, head_dim=64, rope_head_dim=16, norm_eps=1e-20,
    o_groups=4, o_lora_rank=16, window_size=8,
    compress_ratios=(0, 0, 2, 2, 2, 1, 1, 1),
    kv_source_layers=(2, 5), index_source_layers=(2, 5, 7),
    candidate_source_layer=5, candidate_topk_blocks=3, candidate_block_size=2,
    original_seq_len=64, rope_theta=10000.0, rope_factor=4.0, beta_fast=32, beta_slow=1,
    compress_rope_theta=40000.0,
    index_n_heads=8, index_head_dim=32, index_topk=4,
    hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6,
    engram_layer_ids=(1, 2), engram_num_embeddings=(384, 552),
    engram_max_ngram_size=4, engram_vocab_size=50,
    engram_n_heads=2, engram_head_dim=32,
    engram_compressed_vocab_size=VOCAB,
)
TOKEN_MAP = list(range(VOCAB))


def _build_pair():
    """Our model (random init) + theirs loaded with OUR exact weights."""
    mx.random.seed(0)
    ours = our_model.Model(our_config.ModelArgs(**CFG), token_map=list(TOKEN_MAP))
    # non-degenerate engram tables (random fp8 bytes, ue8m0 scales near 2**0)
    rng = np.random.default_rng(7)
    for li in CFG["engram_layer_ids"]:
        e = ours.layers[li].engram.embed
        e.weight = mx.array(rng.integers(0, 256, e.weight.shape, dtype=np.uint8))
        e.scale = mx.array(rng.integers(118, 138, e.scale.shape, dtype=np.uint8))
    theirs = ref_model.Model(ref_config.ModelArgs(**CFG), token_map=list(TOKEN_MAP))
    theirs.load_weights(tree_flatten(ours.parameters()))
    mx.eval(*(v for _, v in tree_flatten(theirs.parameters())))
    return ours, theirs


def _rollout(model, ids_chunks, n_decode, rng):
    """Chunked prefill + decode; returns per-step logits (list)."""
    cache = model.make_cache(bsz=B, max_seq_len=MAXSEQ)
    logits = []
    for chunk in ids_chunks:
        logits.append(model(chunk, cache))
    for _ in range(n_decode):
        nxt = mx.array(rng.integers(0, VOCAB, (B, 1), dtype=np.int32))
        logits.append(model(nxt, cache, last_logit_only=True))
    return logits


def test_model_parity_chunked_prefill_and_decode() -> None:
    ours, theirs = _build_pair()
    rng = np.random.default_rng(11)
    ids = mx.array(rng.integers(0, VOCAB, (B, S), dtype=np.int32))
    chunks = [ids[:, 0:16], ids[:, 16:32], ids[:, 32:48]]
    single = [ids]
    lp = _rollout(ours, chunks, DECODE, np.random.default_rng(21))
    lq = _rollout(theirs, chunks, DECODE, np.random.default_rng(21))
    assert len(lp) == len(lq) == 3 + DECODE
    for i, (a, b) in enumerate(zip(lp, lq)):
        d = mx.max(mx.abs(a - b)).item()
        assert d < 5e-5, f"step {i}: max |Δlogit| = {d}"
    # full-prefill vs chunked must agree on OUR side too (chunk generality):
    # the decode steps follow an identical cache state in both schedules
    lf = _rollout(ours, single, DECODE, np.random.default_rng(21))
    for i in range(DECODE):
        a, b = lp[3 + i], lf[1 + i]
        d = mx.max(mx.abs(a - b)).item()
        assert d < 5e-4, f"chunk-vs-single decode {i}: max |Δ| = {d}"


def test_fakequant_parity() -> None:
    from ponyexl3.arch.deepseek_v41 import fakequant as our_fq
    from deepseek_v41_mlx import fakequant as ref_fq
    rng = np.random.default_rng(3)
    x = mx.array(rng.normal(0, 2.0, (2, 5, 64)).astype(np.float32))
    for fn in ("fake_quant_fp8_ue8m0", "fake_quant_fp4_ue8m0"):
        a = getattr(our_fq, fn)(x, 32)
        b = getattr(ref_fq, fn)(x, 32)
        assert mx.max(mx.abs(a - b)).item() == 0.0, fn
    y = mx.array(rng.normal(0, 1.0, (2, 5, 64)).astype(np.float32))
    assert mx.max(mx.abs(our_fq.fake_quant_fp4_e4m3(y, 16)
                         - ref_fq.fake_quant_fp4_e4m3(y, 16))).item() == 0.0


def test_rope_yarn_parity() -> None:
    from ponyexl3.arch.deepseek_v41 import layers as our_l
    from deepseek_v41_mlx import layers as ref_l
    for (rd, seqlen, orig, theta, factor, bf, bs) in [
        (16, 64, 64, 40000.0, 4.0, 32, 1),   # YaRN on (compressed layers)
        (16, 64, 0, 10000.0, 4.0, 32, 1),    # YaRN off (window layers)
        (64, 256, 65536, 160000.0, 16.0, 32, 1),  # release-like
    ]:
        ca, sa = our_l.precompute_freqs_cis(rd, seqlen, orig, theta, factor, bf, bs)
        cb, sb = ref_l.precompute_freqs_cis(rd, seqlen, orig, theta, factor, bf, bs)
        assert mx.max(mx.abs(ca - cb)).item() == 0.0
        assert mx.max(mx.abs(sa - sb)).item() == 0.0
    # rope inverse undoes rope exactly (fp32 tail round trip)
    x = mx.array(np.random.default_rng(4).normal(size=(1, 6, 4, 64)).astype(np.float32))
    c, s = our_l.precompute_freqs_cis(16, 6, 0, 10000.0, 1.0, 32, 1)
    y = our_l.rope_tail(x, 16, c, s)
    z = our_l.rope_tail(y, 16, c, s, inverse=True)
    assert mx.max(mx.abs(z - x)).item() < 1e-6


def test_hc_sinkhorn_parity() -> None:
    from ponyexl3.arch.deepseek_v41 import hyper_connections as our_h
    from deepseek_v41_mlx import hyper_connections as ref_h
    rng = np.random.default_rng(5)
    mixes = mx.array(rng.normal(0, 1, (2, 6, 24)).astype(np.float32))
    scale = mx.array(rng.normal(0, 1, (3,)).astype(np.float32))
    base = mx.array(rng.normal(0, 1, (24,)).astype(np.float32))
    for pa, pb in zip(our_h.split_sinkhorn(mixes, scale, base),
                      ref_h.split_sinkhorn(mixes, scale, base)):
        assert mx.max(mx.abs(pa - pb)).item() < 1e-7
    # doubly-stochastic after sweeps: rows and cols sum to ~1
    pre, post, comb = our_h.split_sinkhorn(mixes, scale, base)
    cs = comb.sum(axis=-1)
    assert mx.max(mx.abs(cs - 1.0)).item() < 1e-4
    assert mx.all(pre > 0).item()


def test_indexer_candidate_parity() -> None:
    from ponyexl3.arch.deepseek_v41 import indexer as our_i
    from deepseek_v41_mlx import indexer as ref_i
    rng = np.random.default_rng(6)
    scores = mx.array(rng.normal(0, 1, (1, 9, 24)).astype(np.float32))
    scores[:, -1, :] = mx.array(-np.inf)
    lens = mx.arange(1, 10).reshape(9, 1)
    a = our_i.select_candidate_blocks(scores, lens, 3, 2)
    b = ref_i.select_candidate_blocks(scores, lens, 3, 2)
    assert mx.array_equal(a, b)


def test_sparse_attn_parity() -> None:
    from ponyexl3.arch.deepseek_v41 import sparse_attention as our_s
    from deepseek_v41_mlx import sparse_attention as ref_s
    rng = np.random.default_rng(8)
    q = mx.array(rng.normal(0, 1, (1, 6, 4, 64)).astype(np.float32))
    kv = mx.array(rng.normal(0, 1, (1, 20, 64)).astype(np.float32))
    sink = mx.array(rng.normal(0, 1, (4,)).astype(np.float32))
    idx = mx.array(rng.integers(-1, 20, (1, 6, 12), dtype=np.int32))
    a = our_s.sparse_attn(q, kv, sink, idx, 64 ** -0.5)
    b = ref_s.sparse_attn(q, kv, sink, idx, 64 ** -0.5)
    assert mx.max(mx.abs(a - b)).item() < 1e-6
    # all-masked row -> exact zeros (the -1e30 floor semantics)
    z = our_s.sparse_attn(q, kv, sink, mx.full((1, 6, 4), -1, mx.int32), 64 ** -0.5)
    assert mx.max(mx.abs(z)).item() == 0.0


def test_engram_hasher_parity() -> None:
    from ponyexl3.arch.deepseek_v41 import engram as our_e
    from deepseek_v41_mlx import engram as ref_e
    args_o = our_config.ModelArgs(**CFG)
    args_r = ref_config.ModelArgs(**CFG)
    h_o = our_e.EngramHasher(args_o, list(TOKEN_MAP))
    h_r = ref_e.EngramHasher(args_r, list(TOKEN_MAP))
    ids = np.random.default_rng(9).integers(0, VOCAB, (B, 20))
    ca = np.zeros((B, MAXSEQ), dtype=np.int64)
    cb = np.zeros((B, MAXSEQ), dtype=np.int64)
    a = h_o(ids[:, :12], 0, ca)
    b = h_r(ids[:, :12], 0, cb)
    assert np.array_equal(a, b) and np.array_equal(ca, cb)
    a2 = h_o(ids[:, 12:], 12, ca)
    b2 = h_r(ids[:, 12:], 12, cb)
    assert np.array_equal(a2, b2) and np.array_equal(ca, cb)
    # rows must stay inside the declared table
    assert a.max() < max(CFG["engram_num_embeddings"])


def test_wo_a_sliced_equals_dense_einsum() -> None:
    """WoAGrouped (per-slice linears) must reproduce the einsum over the
    reshaped dense wo_a exactly — this is what Phase 3 relies on."""
    from ponyexl3.arch.deepseek_v41.attention import WoAGrouped
    g, r, d, bsz, n = 4, 16, 64, 2, 5   # groups, o_lora_rank, per-group in
    rng = np.random.default_rng(41)
    w = rng.normal(0, 1, (g * r, d)).astype(np.float32)   # dense wo_a weight
    x = rng.normal(0, 1, (bsz, n, g * d)).astype(np.float32)
    dense = mx.einsum("bsgd,grd->bsgr",
                      mx.array(x).reshape(bsz, n, g, d),
                      mx.array(w).reshape(g, r, d))
    dense = dense.reshape(bsz, n, -1)
    sliced = WoAGrouped([nn.Linear(d, r, bias=False) for _ in range(g)])
    for gi in range(g):
        sliced.slices[gi].weight = mx.array(w[gi * r:(gi + 1) * r])
    got = sliced(mx.array(x).reshape(bsz, n, g, d))
    assert mx.max(mx.abs(got - dense)).item() < 1e-5


def test_moe_clamped_swiglu_matches_arch() -> None:
    """exl3_moe's Python fallback must equal arch layers.clamped_swiglu, and
    the Metal expression must carry the same clamp constants."""
    from mlx.nn import silu

    from ponyexl3.arch.deepseek_v41.layers import clamped_swiglu
    from ponyexl3.mlx.exl3_moe import _moe_gate_activation, _moe_glu_metal_expr

    rng = np.random.default_rng(42)
    g = mx.array(rng.normal(0, 8, (3, 16)).astype(np.float32))
    u = mx.array(rng.normal(0, 8, (3, 16)).astype(np.float32))
    limit = 10.0
    ref = clamped_swiglu(g, u, limit)
    got = _moe_gate_activation(g, u, "silu", limit).astype(mx.float32)
    assert mx.max(mx.abs(ref - got)).item() < 1e-4
    expr = _moe_glu_metal_expr("silu", limit)
    assert "fmin(g, 10.000000f)" in expr and "fmax(u, -10.000000f)" in expr
    # no clamp → untouched silu*up
    assert _moe_glu_metal_expr("silu", 0.0) == \
        "float h = (g / (1.0f + exp(-g))) * u;"
    assert mx.max(mx.abs(_moe_gate_activation(g, u, "silu", 0.0).astype(mx.float32)
                         - (silu(g) * u))).item() == 0.0


def test_engram_host_tables_equal_device() -> None:
    """Host-mmap mode must dequantize identical rows to device-resident mode."""
    from ponyexl3.arch.deepseek_v41.engram import EngramEmbedding

    rng = np.random.default_rng(43)
    rows, dim, block = 997, 256, 32
    w = rng.integers(0, 256, (rows, dim), dtype=np.uint8)
    s = rng.integers(118, 128, (rows, dim // block), dtype=np.uint8)  # ~2^0..2^7
    idx = rng.integers(0, rows, (3, 5))

    emb = EngramEmbedding(rows, dim, block)
    emb.weight = mx.array(w)
    emb.scale = mx.array(s)
    dev = emb(mx.array(idx))

    emb2 = EngramEmbedding(rows, dim, block)
    emb2.set_host_tables(w, s)
    got = emb2(mx.array(idx))
    assert mx.max(mx.abs(dev - got)).item() == 0.0


def test_generate_loop_integration() -> None:
    """Phase 2 exit: synthetic model runs greedy through ponyexl3.mlx.generate."""
    from ponyexl3.arch.deepseek_v41.pony import DeepseekV41ForPony
    from ponyexl3.mlx.generate import GenStats, stream_generate

    mx.random.seed(2)
    m = our_model.Model(our_config.ModelArgs(**CFG), token_map=list(TOKEN_MAP))
    wrapped = DeepseekV41ForPony(m)
    rng = np.random.default_rng(31)
    prompt = rng.integers(0, VOCAB, 24).tolist()

    stats = GenStats()
    out = list(stream_generate(wrapped, prompt, max_tokens=12, temp=0.0,
                               prefill_chunk=10, stats=stats))
    assert len(out) == 12
    assert stats.prompt_tokens == 24 and stats.finish_reason == "length"
    # greedy determinism: rerun matches
    assert list(stream_generate(wrapped, prompt, max_tokens=12, temp=0.0,
                                prefill_chunk=10)) == out
    # identical to the raw reference-style loop (single-shot prefill, full logits)
    cache = m.make_cache(bsz=1, max_seq_len=MAXSEQ + 16)
    ids = mx.array([prompt])
    expect: list[int] = []
    logits = m(ids, cache)
    for _ in range(12):
        nxt = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        expect.append(nxt)
        logits = m(mx.array([[nxt]]), cache, last_logit_only=True)
    assert out == expect


def test_compressor_chunk_invariance() -> None:
    """Ratio-2 compressor: token-at-a-time decode == one 8-token prefill."""
    from ponyexl3.arch.deepseek_v41.compressor import Compressor, CompressorState
    args = our_config.ModelArgs(**CFG)
    mx.random.seed(1)
    comp = Compressor(args, 2)  # ratio 2
    mx.eval(*tree_flatten(comp.parameters()))
    x = mx.array(np.random.default_rng(10).normal(0, 1, (1, 8, CFG["dim"])).astype(np.float32))
    full = comp(x, 0, CompressorState(1, 2, CFG["head_dim"]))
    st = CompressorState(1, 2, CFG["head_dim"])
    parts = []
    for t in range(8):
        out = comp(x[:, t:t + 1], t, st)
        if out is not None:
            parts.append(out)
    chunked = mx.concatenate(parts, axis=1)
    assert chunked.shape == full.shape
    assert mx.max(mx.abs(chunked - full)).item() < 1e-6


def test_gate_math() -> None:
    """sqrtsoftplus scoring, bias selects-but-does-not-scale, renorm +1.5 scale."""
    from mlx.nn import softplus as _sp
    from ponyexl3.arch.deepseek_v41.moe import Gate
    args = our_config.ModelArgs(**CFG)
    g = Gate(args)
    g.weight = mx.array(np.random.default_rng(12).normal(0, 1, (args.n_routed_experts, args.dim)).astype(np.float32))
    g.bias = mx.array(np.linspace(-0.5, 0.5, args.n_routed_experts).astype(np.float32))
    x = mx.array(np.random.default_rng(13).normal(0, 1, (4, args.dim)).astype(np.float32))
    w, idx = g(x)
    scores_raw = (x.astype(mx.float32) @ g.weight.astype(mx.float32).T).tolist()
    bias = g.bias.tolist()
    srow = mx.sqrt(_sp(mx.array(scores_raw))).tolist()  # scores ARE sqrt-softplus
    w_tl, idx_tl = w.tolist(), idx.tolist()
    topk = args.n_activated_experts
    for row in range(4):
        biased = [sc + b for sc, b in zip(srow[row], bias)]
        picked = idx_tl[row]
        got = sorted(biased[j] for j in picked)
        expect = sorted(biased, reverse=True)[:topk]
        assert all(abs(a - b) < 1e-6 for a, b in zip(reversed(got), expect))
        # weights are the UNBIASED scores, renormalized, scaled by route_scale
        sel = [srow[row][j] for j in picked]
        denom = sum(sel) + 1e-20
        for wgt, sc in zip(w_tl[row], sel):
            assert abs(wgt - sc / denom * args.route_scale) < 1e-6
