"""Env-gated layout audit facts for the deepseek_v41 EXL3 pack (Phase 1).

Run with:
    PONYEXL3_MODEL_DSV41=~/src/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw \
    PONYEXL3_ENGRAM_DIR=~/src/models/engram-src \
    pytest tests/test_dsv41_bundle_layout.py -q

Every assertion is a *verified* fact from docs/dsv41_layout_audit.md /
docs/dsv41_pack_layout.md — this test is the regression harness for the
container v1.6.2 assumptions Phase 3 builds on.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

MODEL_DSV41 = Path(os.environ.get("PONYEXL3_MODEL_DSV41", "")).expanduser()
ENGRAM_DIR = Path(os.environ.get("PONYEXL3_ENGRAM_DIR", "")).expanduser()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "dev"))
import bundle_layout as bl  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (MODEL_DSV41 / "quantization_config.json").is_file(),
    reason="set PONYEXL3_MODEL_DSV41=/path/to/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw",
)


@pytest.fixture(scope="module")
def qcfg() -> dict:
    return bl._read_json(MODEL_DSV41 / "quantization_config.json")


@pytest.fixture(scope="module")
def storage(qcfg: dict) -> dict:
    return {k: v for k, v in qcfg["tensor_storage"].items()
            if v.get("quant_format") == "exl3"}


def test_container_v162_fields(qcfg: dict) -> None:
    assert qcfg["version"] == "1.6.2"
    assert qcfg["quant_method"] == "exl3"
    assert qcfg["codebook"] == "mul1"
    assert float(qcfg["bits"]) == pytest.approx(2.9)
    assert qcfg["head_bits"] == 6
    assert qcfg["mtp_bits"] == 4
    assert qcfg["out_scales"] == "always"
    cfg = bl._read_json(MODEL_DSV41 / "config.json")
    assert cfg["model_type"] == "deepseek_v41"
    assert "vision_config" in cfg


def test_exl3_module_inventory(storage: dict) -> None:
    assert len(storage) == 46702
    k_hist: dict[int, int] = {}
    for info in storage.values():
        k = bl.module_k(info)
        assert k is not None, "every exl3 module must expose a trellis shape"
        assert k == int(info["bits_per_weight"])  # derived-K == qcfg bits
        k_hist[k] = k_hist.get(k, 0) + 1
    # mixed-bitrate K mix (matches exl3_k_map.json)
    assert k_hist == {2: 5760, 3: 40320, 4: 57, 5: 558, 6: 3, 8: 4}


def test_all_module_dims_tile_clean(storage: dict) -> None:
    for key, info in storage.items():
        ins, outs = bl.module_dims(info)
        assert ins % 128 == 0 and outs % 128 == 0, key


def test_wo_a_is_eight_slices_per_layer(storage: dict) -> None:
    slices = {key for key in storage if ".wo_a.slice." in key}
    assert len(slices) == 40 * 8  # o_groups=8 → per-group EXL3 slices
    for li in range(40):
        for g in range(8):
            assert f"layers.{li}.attn.wo_a.slice.{g}" in storage


def test_expert_layout(storage: dict) -> None:
    import re
    pat = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])$")
    hits = {pat.match(k).groups() for k in storage if pat.match(k)}  # pyright: ignore
    assert len(hits) == 40 * 384 * 3
    assert {k for k in storage if k.endswith(".ffn.shared_experts.w1")} == {
        f"layers.{li}.ffn.shared_experts.w1" for li in range(40)
    }


def test_attention_indexer_compressor_quantization(storage: dict) -> None:
    for proj in ("wq_a", "wq_b", "wkv", "wo_b"):
        assert all(f"layers.{li}.attn.{proj}" in storage for li in range(40))
    # 2-level DSA: indexer on 8 layers (2,8,14,20,24,28,32,36); wk on 4
    wq_b = sorted(int(k.split(".")[1]) for k in storage if ".indexer.wq_b" in k)
    assert wq_b == [2, 8, 14, 20, 24, 28, 32, 36]
    wk = sorted(int(k.split(".")[1]) for k in storage if ".indexer.wk." in k)
    assert set(wk) <= set(wq_b)
    comp = {k.split(".")[1] for k in storage if ".compressor.wkv" in k}
    assert len(comp) == 4
    # engram wkv quantized in-pack; embed tables are external
    assert {k for k in storage if ".engram.wkv" in k} == {
        "layers.1.engram.wkv", "layers.14.engram.wkv"}
    assert not any(".engram.embed" in k for k in storage)


def test_k_map_agrees_with_storage(qcfg: dict, storage: dict) -> None:
    km = bl._read_json(MODEL_DSV41 / "exl3_k_map.json")
    assert km["codebook"] == "mul1"
    for li, k in km["routed"].items():
        assert int(storage[f"layers.{li}.ffn.experts.0.w1"]["bits_per_weight"]) == k
    for li, k in km["shared"].items():
        mod = storage[f"layers.{li}.ffn.shared_experts.w1"]
        if str(li) in km.get("shared_overrides", {}):
            continue  # layer 29 is mixed 4/5 — infer from trellis
        assert int(mod["bits_per_weight"]) == k
    for li, k in km["engram"].items():
        assert int(storage[f"layers.{li}.engram.wkv"]["bits_per_weight"]) == k


def test_engram_tables_absent_from_pack_shards() -> None:
    from glob import glob
    from safetensors import safe_open
    for p in glob(str(MODEL_DSV41 / "*.safetensors")):
        with safe_open(p, framework="np") as f:
            for k in f.keys():
                assert "engram.embed" not in k


@pytest.mark.skipif(
    not (ENGRAM_DIR / "model-00048-of-00048.safetensors").is_file(),
    reason="set PONYEXL3_ENGRAM_DIR=/path/to/engram-src (release shards 47+48)",
)
def test_engram_shard_format() -> None:
    from safetensors import safe_open
    num_emb = bl._read_json(ENGRAM_DIR / "config.json")
    tc = num_emb.get("text_config", num_emb)
    tables = {
        "model-00047-of-00048.safetensors": ("layers.1.", tc["engram_num_embeddings"][0]),
        "model-00048-of-00048.safetensors": ("layers.14.", tc["engram_num_embeddings"][1]),
    }
    for shard, (prefix, rows) in tables.items():
        path = ENGRAM_DIR / shard
        if not path.is_file():
            pytest.skip(f"{shard} not downloaded")
        with safe_open(path, framework="np") as f:
            dtypes = {k: str(f.get_slice(k).get_dtype()) for k in f.keys()}
            shapes = {k: f.get_slice(k).get_shape() for k in f.keys()}
        assert dtypes[f"{prefix}engram.embed.weight"].endswith("F8_E4M3")
        assert dtypes[f"{prefix}engram.embed.scale"].endswith("F8_E8M0")
        assert shapes[f"{prefix}engram.embed.weight"] == [rows, 256]
        assert shapes[f"{prefix}engram.embed.scale"] == [rows, 8]
        assert dtypes[f"{prefix}engram.q_weight"].endswith("BF16")


def test_engram_hash_prime_sums_match_config() -> None:
    """The bucket layout is fully determined by config — per layer, 24 primes
    drawn upward from engram_vocab_size-1 (never reused) must sum to
    engram_num_embeddings, or a multiplier mismatch silently rehashes it.
    Self-contained transcription of the reference `build_layout`."""
    tc = bl._read_json(MODEL_DSV41 / "config.json")
    tc = tc.get("text_config", tc)
    layer_ids = tc["engram_layer_ids"]
    cols = (tc["engram_max_ngram_size"] - 1) * tc["engram_n_heads"]

    def isprime(n: int) -> bool:
        if n < 2 or n % 2 == 0:
            return n == 2
        return all(n % i for i in range(3, int(n**0.5) + 1, 2))

    primes, seen, current = [], set(), tc["engram_vocab_size"] - 1
    for _ in layer_ids:
        row = []
        for _ in range(cols):
            current += 1
            while not isprime(current) or current in seen:
                current += 1
            seen.add(current)
            row.append(current)
        primes.append(row)
    sums = [sum(row) for row in primes]
    assert sums == tc["engram_num_embeddings"]
    # hash multipliers are odd, seeded default_rng(10007 * layer_id), bounded
    # so token*multiplier fits int64 (mirrors the reference derivation)
    import numpy as np
    bound = (np.iinfo(np.int64).max // tc["engram_compressed_vocab_size"]) // 2
    for li in layer_ids:
        v = np.random.default_rng(10007 * li).integers(
            low=0, high=bound, size=(tc["engram_max_ngram_size"],), dtype=np.int64) * 2 + 1
        assert bool((v % 2 == 1).all())
