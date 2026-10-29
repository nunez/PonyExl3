# deepseek_v41 development log

Chronological record for the `deepseek_v41` effort. Plan:
[`deepseek_v41_plan.md`](deepseek_v41_plan.md). Every measurement carries
machine + commit + context tags.

Machine of record: **Mac Studio, Apple M3 Ultra, 512 GB unified memory**,
macOS. Fork: `nunez/PonyExl3`, base upstream `beamivalice/PonyExl3` @ `8e7fa6b`
(v0.3.0).

---

## 2026-10-29 — Phase 0 start

- Fork created (`nunez/PonyExl3`, default branch `master`); remotes:
  `origin` = fork, `upstream` = `beamivalice/PonyExl3`.
- Plan committed (`docs/deepseek_v41_plan.md`).
- Environment: `uv venv --python 3.14` (system Python is 3.15.2 — pinning to
  the upstream-tested 3.14 line to avoid mlx wheel gaps),
  `uv pip install -e ".[dev]"`, then full `pytest tests/ -q` baseline on the
  M3 Ultra. Results appended below when green.
- Started download: `turboderp/Qwen3.8-27B-exl3` rev `4.00bpw` — the modern
  container canary (exl3 **v1.4.2**, `mul1` codebook, `mtp_bits: 4`) that
  gates Phase 3's confidence in the DS pack's **v1.6.2** container.

### Baseline pytest (M3 Ultra)

- `uv sync --extra dev` then `pytest tests/ -q` → **290 passed, 22 skipped**
  (22 = model-dir-gated integration tests), 50 s. mlx **0.31.2**,
  mlx-lm **0.31.3** per `uv.lock`.
- ⚠️ **Known risk:** with unpinned latest mlx **0.32.4**,
  `test_native.py::test_fused_swiglu_matches_unfused` **fails** (max abs err
  0.125 vs 1e-2 gate) — fused SwiGLU kernel behavior changed upstream in
  0.32.x. All development uses the locked env (`uv sync`); treat any
  mlx ≥0.32 bump as a re-validation gate for every fused kernel.

### Qwen3.8-27B 4.00bpw smoke (container v1.4.2 canary)

**Phase 0 GO.** Bundle: `models/Qwen3.8-27B-exl3-4.00bpw` (turboderp rev
`4.00bpw`, 16 GB, 2 shards, qcfg `version: 1.4.2`, `codebook: mul1`,
`mtp_bits: 4`, 707 `tensor_storage` entries, `model_type: qwen3_5` VL-wrapped).

- `ponyexl3-generate --mtp off`: load **7.2 s**, 129 exact EXL3 linears
  (65 huge/GEMV), coherent greedy output. **14.2 tok/s** decode (58-token
  prompt, 128 gen), prefill 30 tok/s (short-prompt, cold).
- `--mtp auto`: bundled head auto-discovered (`[mtp] draft head loaded`),
  verify-gated speculation active — 70/174 drafts accepted, 2.21 tok/cycle,
  **16.7 tok/s**. (MTP win is modest here; draft acceptance on this prompt
  class is 40% — revisit with `--dflash`-style block drafter later.)
- Consistency vs README curve: sits between M3 Max (11.3) and M5 Max (16.6)
  plain decode — plausible for M3 Ultra on their kernels.
- **Conclusion:** PonyExl3 loads exl3 **v1.4.2 + mul1 + bundled MTP** natively,
  no loader patches needed → Phase 3's v1.6.2 DS pack is on the table.

### Phase 1 tooling dry-run: `tools/dev/bundle_layout.py` on Qwen3.8 canary

- Shard-**header**-only audit tool (no weight materialization → seconds even
  on the 43-shard DS pack). Dry-run on the canary:
  - 401 EXL3 modules in `tensor_storage` (1 `lm_head` + 400 under
    `model.language_model.*`); loader's "[load] 129 exact EXL3 linears" is
    **post-sibling-fusion** tree count, and strict-load with zero
    `skipped_unmapped` warnings proves all 401 map to slots.
  - K histogram empty at trellis-`shape` level: turboderp's qcfg stores
    trellis shapes per entry differently than assumed for K-histogram —
    **fix `module_k()` when we first run it on the DS pack (Phase 1)**;
    dims/divisibility/plain-inventory paths all work.
  - Plain inventory: 2.42 GiB text (norms/embeds/etc) + 0.20 GiB `mtp.*` +
    0.86 GiB `vision.*` — matches the expected VL-wrapper + MTP bundle shape.

---

## 2026-10-29 — Phase 1 start: disk, tooling fixes, source-side audit


- **Disk:** freed ~1.5 TB (retired `.mtplx`/`.omlx` runtimes, H3 BF16
  weights, kapitsa/copilot caches) → 1.6 TB free on the 4 TB volume.
  Downloads: LibertAI EXL3 pack (~205 GB) → `~/src/models/DeepSeek-V4.1-
  Flash-1017-EXL3-2.9bpw`; DeepSeek shards 47/48 = Engram tables
  (2 × 101.5 GB) + index + config → `~/src/models/engram-src`.
- `tools/dev/bundle_layout.py` **fixed** (devlog Phase 0 follow-up):
  qcfg `stored_tensors` entries are keyed by the fully qualified tensor
  name (`"<module>.trellis"`), not the bare suffix — `module_k()` now
  mirrors the loader's lookup; `module_dims()` derives in/out from trellis
  tiles (×16) since qcfg has no `input_features` fields. Canary re-run:
  K histogram populates (400×K4 + lm_head K6 == `head_bits: 6`),
  derived-K vs qcfg `bits_per_weight` mismatches: 0.
- Cloned `PipeNetwork/deepseek-v41-mlx` (12 MB) → `~/src/deepseek-v41-mlx`:
  port source + parity oracle + torch reference (`docs/reference/`) + the
  release's full safetensors index (`hub_index.json`).
- **Source-side key inventory verified** (96,085 tensors): inference-style
  names, no `model.` prefix (`layers.N.attn.wq_a.weight`, `ffn.experts.N.w1`,
  `embed/head`) → Phase 3 needs an explicit remap table, see
  `docs/dsv41_layout_audit.md` §1/§5. DSpark = `mtp.0..2`; vision = 266
  tensors; engram tables live only in shards 47/48.
- **Engram hash math verified from config alone:** 24 sequential primes/layer
  from 16,000,000−1 sum exactly to `engram_num_embeddings`
  `[384006168, 384016682]` ✔; multipliers odd, from `default_rng(10007·layer)`
  bounded by `int64max/99092/2`. Compressed-vocab rebuild must equal 99,092
  (hard error on mismatch — we adopt the same guard).

---

## 2026-10-29 — Phase 1 pack-side verification (pack + engram landed)

- Downloads: EXL3 pack **196 GB rc=0** (43 shards) → `~/src/models/DeepSeek-
  V4.1-Flash-1017-EXL3-2.9bpw`; engram shard 48 + index + config landed;
  shard 47 re-downloading (first `hf download` call silently dropped one of
  four explicit filenames — re-invoked per-file).
- `docs/dsv41_pack_layout.md` generated with the fixed auditor: container
  **v1.6.2** loads its qcfg natively; **46,702 EXL3 modules**; K histogram
  {2:5760, 3:40320, 4:57, 5:558, 6:3, 8:4} == `exl3_k_map.json` (80/80 spot
  checks); **every module's dims %128-clean — Phase 1 tile gate PASSES**.
- **Container field correction:** the pack's qcfg uses `bits: 2.9`, not
  `avg_bits` (plan §1 note was wrong — no loader tolerance needed beyond
  float `bits`).
- New structural findings:
  - `wo_a` is quantized as **8 per-o-group slices per layer** (320 modules)
    — a better answer than PipeNetwork's never-quantize refusal.
  - Indexers on 8 layers (`wq_b`), `indexer.wk` on 4; `compressor.*`
    quantized (K5) though PipeNetwork refuses; `engram.wkv` EXL3 (K5/K4);
    embed plain bf16, head K6 == `head_bits`.
  - qcfg carries **empty-`stored_tensors` placeholder entries**
    (`hc_expand`, `layers.N.hc_{attn,ffn}.norm`) — loader must skip.
  - `mtp.*` bundled **plain** (4839 tensors, 6.88 GiB) despite `mtp_bits: 4`
    — Phase 6b may need the FP8 dequant path, not EXL3, for DSpark.
  - Pack shards contain **zero `engram.embed` keys** → tables come from
    `engram-src` (verified: F8_E4M3 [384M×256] + F8_E8M0 [384M×8], direct
    mmap possible).
- `tests/test_dsv41_bundle_layout.py` (env-gated, 10 tests) pins every fact
  above as regression harness: **9 passed, 1 skipped** (shard 47 pending);
  full suite 290 passed / 32 skipped.
- **Phase 1 exit criteria met** (47's header check auto-enables when it
  lands). Phase 2 (pure-MLX arch) unblocked.

---

## 2026-10-29 — Phase 2: pure-MLX deepseek_v41 arch + parity

- New package `ponyexl3/arch/deepseek_v41/` (14 modules, ~2.1k lines):
  name-for-name port of PipeNetwork/deepseek-v41-mlx (MIT; attributed per
  file). Module tree == checkpoint tensor names (`layers.N.attn.wq_a`,
  `ffn.experts.N.w1`, `embed/head`) → Phase 3 EXL3 swap is 1:1. Zero
  mlx_lm-arch dependency (mlx-lm still lacks the arch, #1233).
- Covers: YaRN rope (+inverse, adjacent-pair, tail-only), QAT fake-quant
  triplets (fp8-e4m3/ue8m0 window KV @32, fp4-e2m1/e4m3 compressed KV @16,
  fp4/ue8m0 indexer q/k @32 — bit-exact vs oracle incl. IEEE-bit scale
  ceil and ties-to-even e2m1 grid), gather-based sparse attn (shared
  MLA kv, attn sink in denominator only, -1e30 all-masked zero rows),
  chunk-general softmax-gate compressor (ratio>1 pooled / ratio==1 plain),
  2-level indexer (candidate blocks pinned newest, ReLU + weights_proj
  collapse, top-512 sorted), mHC staggered Sinkhorn (comb on summed-axis
  j), sqrtsoftplus gate (bias selects-not-scales, 1e-20 renorm, ×1.5),
  clamped SwiGLU(10) MoE (SwitchGLU gather), Engram hasher (prime buckets,
  odd multipliers, compressed map) + fp8 row-gather embedding, and the
  staggered-hc CED-style block stack.
- Cache classes: window ring (chrono read + wrapped write), compressed-KV
  append buffers with consumer references, compressor partial-group state,
  index-key cache, per-forward SharedState (kv/index/topk/candidates).
- **Parity harness** `tests/test_dsv41_arch_parity.py`
  (PONYEXL3_DSV41_MLX-gated, 10 tests): our random-init weights transplanted
  into the oracle's identically-named tree; **identical logits (<5e-5) over
  3-chunk prefill + 8 decode steps**; fakequant/rope/sinkhorn/candidate-
  block/sparse-attn/engram-hash submodule parity exact; compressor
  chunk-invariance; rope-inverse round-trip; gate-math invariants;
  chunk-vs-single decode identity.
- **Generate-loop integration**: `Model.hidden()` (mlx_lm-style inner path)
  + `DeepseekV41ForPony` view (`.model`/`.lm_head`/`make_cache`);
  `stream_generate` greedy on the tiny model == raw reference loop,
  deterministic across runs. Note: no explicit encoder/decoder prefill
  split exists in the reference — CED is fully expressed via
  `compress_ratios`/`kv_source_layers`/`index_source_layers` + SharedState.
- Suite: 300 passed / 32 skipped. Phase 2 exit met; Phase 3 (EXL3
  integration) unblocked — mapping table is audit §5.
