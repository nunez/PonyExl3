# deepseek_v41 layout audit (Phase 1) — PACK VERIFIED 2026-10-29

Source-side facts are **verified** from the release's safetensors index
(`deepseek-v41-mlx/docs/reference/hub_index.json`, 96,085 tensors,
510.3 GB `total_size`) and `config.json`. Pack-side facts verified from
`~/src/models/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` (196 GB, 43 shards,
rc=0 download); full machine-generated report in
[`dsv41_pack_layout.md`](dsv41_pack_layout.md), regression harness in
`tests/test_dsv41_bundle_layout.py` (env-gated).

## 1. Release (source) key inventory — verified

Naming is **inference-style**: no `model.` prefix, no `self_attn`/`mlp`/
`weight_scale_inv`; `ffn` not `mlp`; experts are `w1/w2/w3` not
`gate/up/down_proj`; lm_head is `head.weight`, embedding is `embed.weight`.
This matches neither mlx_lm's tree nor PonyExl3's `language_model`-wrapper
expectations → Phase 3 needs an explicit key-remap table (see §5).

| Template | ×# | Notes |
|---|---|---|
| `embed.weight`, `head.weight` | 1 | [129280, 5120] both |
| `layers.N.attn.{wq_a,wq_b,wkv,wo_a,wo_b}.weight` + `.scale` | 40 each | fp8+ue8m0 32×32 blocks |
| `layers.N.attn.{q_norm,k_norm}.weight`, `attn.attn_sink` | 40 | plain |
| `layers.N.attn.indexer.{wq_b.weight+scale, wk.weight, weights_proj.weight, k_norm.weight}` | 8 / 4 / 4 / 4 | `indexer.wq_b` is **already fp8 in the release** |
| `layers.N.attn.compressor.{wkv.weight+scale, wgate.weight, norm.weight}` | 4 / 3 / 4 | tiny |
| `layers.N.{attn_norm,ffn_norm}.weight` | 40 | plain |
| `layers.N.hc_{attn,ffn}_{base,fn,scale}` | 40 each | mHC, fp32 |
| `layers.N.ffn.experts.N.{w1,w2,w3}.weight+scale` | 15,360 = 40·384·3 | **fp4** packed nibbles + per-32 ue8m0 |
| `layers.N.ffn.shared_experts.{w1,w2,w3}.weight+scale` | 40 each | |
| `layers.N.ffn.gate.{weight,bias,bias_vl}` | 40 | router, fp32 |
| `layers.{1,14}.engram.embed.{weight,scale}` | 1 | uint8 fp8-e4m3 [~384 M, 256] + ue8m0 [~384 M, 8] |
| `layers.{1,14}.engram.{q_weight,k_weight,wkv.weight+scale}` | 1 | gating params |
| `mtp.{0,1,2}.*` | ~700 | DSpark stack: same per-layer layout + `main_proj.{weight,scale}`, `main_norm`, 128 experts × 3 |
| `vision.*`, `aligner.*`, `image_{start,end,newline}` | 266 + 4 | DeepSeek-ViT tower |

Engram tables are physically isolated: `layers.1.engram.*` →
`model-00047-of-00048.safetensors`, `layers.14.engram.*` → shard 48, 101.5 GB
each — the two files we mirror into `~/src/models/engram-src/`.

## 2. Tile constraints (derived from config, all %16 ✓ / %128 where relevant)

`hidden_size 5120` · `q_lora_rank 1280` · `o_lora_rank 1024` · `head_dim 512`
· `index_n_heads·index_head_dim = 4096` · `moe_intermediate_size 2304`
(= 16·144 = 128·18) · `engram wkv in 24·256 = 6144 / out 5·5120 = 25600` ·
`vocab 129280` (= 128·1010). No odd dims found at the projection level; the
final per-module verdict comes from `bundle_layout.py` on the pack
(`odd16`/`odd128` sections).

## 3. Engram hashing — verified against release config

Re-derived the table layout from config alone (via
`deepseek_v41_mlx.engram.build_layout`): 24 disjoint primes per layer
(3 n-gram sizes × 8 heads) drawn upward from `engram_vocab_size − 1` sum to
exactly `engram_num_embeddings = [384006168, 384016682]` ✔ — the hash bucket
math is fully determined by `(vocab 16 M, max_ngram 4, heads 8, layers [1,14])`
plus per-layer multipliers from `default_rng(10007·layer_id)` bounded by
`int64max / compressed_vocab 99092 / 2`. Compressed-token map must rebuild to
exactly 99,092 classes from the pack's tokenizer (`encoding/` assets), else
the table silently rehashes (PipeNetwork raises on mismatch — we will too).

Table storage: weight = uint8 fp8-e4m3 rows [R, 256]; scale = uint8 ue8m0
[R, 8] (block 32). Read-only gather → our Phase 3 path is mmap + row gather +
fp8 dequant kernel, **not** EXL3 (LibertAI keeps them plain too — see §4).

## 4. EXL3 pack layout — VERIFIED

Container: `version: 1.6.2`, `bits: 2.9` (top-level field is **`bits`**, not
`avg_bits` as the plan note said), `codebook: mul1`, `head_bits: 6`,
`mtp_bits: 4`, `out_scales: always`. `exl3_k_map.json` publishes the K mix
and matches `tensor_storage` exactly.

**46,702 EXL3 modules**, derived-K == qcfg `bits_per_weight` on all of them;
K histogram `{K2: 5760, K3: 40320, K4: 57, K5: 558, K6: 3, K8: 4}`;
**every module's in/out dims are %128-clean** (the Phase 1 tile gate).

Per-template (×40 layers unless noted):

| Module | K | Notes |
|---|---|---|
| `layers.N.ffn.experts.N.w{1,2,3}` (384·3) | 3 (layers 18–22 → 2) | 46,080 modules |
| `layers.N.ffn.shared_experts.w{1,2,3}` | 5 (11–20 → 4; layer 29 mixed, infer from trellis) | |
| `attn.wq_a`, `attn.wkv` | 6 | |
| `attn.wq_b`, `attn.wo_b` | 5 | |
| **`attn.wo_a.slice.0..7`** | 5 | **LibertAI splits `wo_a` into `o_groups=8` per-group EXL3 slices** — sidesteps PipeNetwork's "block-diagonal reshape carves packed bytes" ban; loader must present the 8 slices as one logical `wo_a` (or run 8 grouped GEMVs) |
| `indexer.wq_b` (8 layers: 2,8,14,20,24,28,32,36) | 5 | 2-level DSA indexers on every 4th layer |
| `indexer.wk` (4 layers) | 8 | |
| `compressor.wkv` (4 layers), `compressor.wgate` (3 layers) | 5 | LibertAI quantizes what PipeNetwork refuses (`compressor.*`); `wgate` absent on 1 layer |
| `engram.wkv` (layers 1, 14) | 5 / 4 | small projection is EXL3; **embed tables external** (0 `engram.embed` keys in-pack) |
| `head` | 6 | == `head_bits`; `embed` plain bf16 (1.23 GiB) |

Plain tensors (9.31 GiB total, shard-header scan): norms/sinks/`hc_*_{base,
fn,scale}`/router `gate.{weight,bias,bias_vl}`/`indexer.k_norm`/
`compressor.norm`/`embed`/`norm` + `mtp.*` 4839 tensors (6.88 GiB, plain —
loader skips) + `vision.*`/`aligner.*`/`image_*` (~0.9 GiB, skipped).
Quirk: qcfg also carries **empty `stored_tensors: {}` placeholders**
(`hc_expand`, `layers.N.hc_{attn,ffn}.norm`) that have no shard tensors —
loader must tolerate entries with no stored data.

**Engram source tables** (`~/src/models/engram-src` = release shards
47/48): `embed.weight` `F8_E4M3` [384,016,682 × 256] + `embed.scale`
`F8_E8M0` [384,016,682 × 8] (block 32) — native FP8 dtypes in safetensors,
direct `mmap` byte access works; `q/k_weight` bf16 [4, 5120]; `wkv` fp8
(we use the pack's EXL3 `engram.wkv` instead).

## 5. Loader gaps to close in Phase 3

1. `_ARCHITECTURES`: add `deepseek_v41` / `deepseek_v41_text` → local
   `ponyexl3/arch/deepseek_v41` (no `mlx_lm` module exists yet).
2. Key remap: release keys have **no `model.` prefix and no
   `language_model` wrapper**; `_module_path()` must special-case
   `embed.weight` → `model.embed_tokens`-equivalent, `head.weight` → lm_head,
   `layers.N.*` passthrough, and expert regex must be
   `layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])` (not Qwen's
   `mlp.experts.…gate_proj`).
3. Expert stacking names `w1/w2/w3` → `EXL3SwitchGLU` gate/up/down order =
   `w1/w3/w2` mapping (reference converter: w1=gate, w3=up, w2=down).
4. qcfg tolerations: version `1.6.2`, top-level `bits` (float 2.9 — the plan's
   `avg_bits` note was wrong), `mtp_bits`, `out_scales: always`; and entries
   with **empty `stored_tensors`** (`hc_expand`, `hc_*.norm`) must be ignored.
5. `wo_a` arrives as `attn.wo_a.slice.{0..7}` EXL3 modules — build as 8
   EXL3Linears (one per o-group) or fuse into one logical projection.
6. Engram tables: keep plain uint8/uint8, `np.memmap` → device gather; never
   build host fp16 copies (Phase 4 policy).

## 6. Memory math (M3 Ultra 512 GB)

| Resident | GB |
|---|---|
| EXL3 2.9 bpw shards | ~205 |
| Engram fp8 tables (both) | ~194 (2 × 101.5) |
| KV @ 8k ctx (CSA2 compressed + SWA + indexer) | ≪ 10 |
| **Total vs ~400 GB wired budget** | **~400 — tight**; Engram mmap-lazy default per Phase 4 |
