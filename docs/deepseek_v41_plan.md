# deepseek_v41 on PonyExl3 — development plan

**Status: Phase 0 ✅ GO (2026-10-29)** — see
[`dsv41_devlog.md`](dsv41_devlog.md) for measurements.

**Goal:** run `deepseek_v41` (DeepSeek-V4.1-Flash-1017) EXL3 checkpoints
text-only on Apple Silicon through `ponyexl3-generate`, starting from the
ungated `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` pack.
`BlackwellBoy/…-BLACKFOLD-EXL3-2.9bpw-TP2` is the same weights + a 152-tensor
`wo_b` refusal patch overlay, so everything built here applies to it verbatim.

**Target hardware:** Mac Studio M3 Ultra, 512 GB (resident budget ~400 GB:
~210 GB EXL3 shards + ~189 GB Engram + KV + headroom).

**Repo:** fork `nunez/PonyExl3` (default branch `master`, upstream
`beamivalice/PonyExl3`). We push to `origin`; upstream contribution PRs come
at the end (Phase 7).

**Why this is worth doing:** at 2.9 bpw with 16 B active params/decode token,
weight traffic is ~6–7 GB/token. M3 Ultra ≈ 800 GB/s → a frontier-class 552 B
model should decode at tens of tok/s on a desktop. No EXL3 runtime for
`deepseek_v41` exists on macOS today; the only Mac runtimes are separate MLX
ports (`deepseek-v41-mlx`, gbx-lm, oMLX). `mlx-lm` itself still lacks the arch
(ml-explore/mlx-lm#1233).

---

## 1. Facts this plan is built on (verified Oct 2026)

### Checkpoint / container
- `model_type: deepseek_v41`, text `deepseek_v41_text`, arch
  `DeepseekV41ForCausalLM`; `vision_config` present (DeepSeek-ViT); text-only
  runtime drops it.
- EXL3 container **v1.6.2**, `codebook: mul1`, `avg_bits: 2.9` (note:
  `avg_bits`, not `bits` — PonyExl3 must tolerate this), `head_bits: 6`,
  `mtp_bits: 4` (DSpark/MTP head bundled), `out_scales: always`.
- `exl3_k_map.json` (1.4 KB) publishes the per-module K mix;
  `quantization_config.json` is a 54 MB per-module storage map. PonyExl3's
  loader already derives K per module from trellis shapes → mixed-bitrate
  bundles are natively fine.
- **Engram tables are external**: `engram_table_dir` is null in-repo; the
  BlackwellBoy/lucataco recipes point at a separate `engram-src` folder
  (DeepSeek's own Engram shards 47/48, FP8, ~189 GB). `encoding/` dir in the
  pack holds the DSML tokenizer assets.
- Original quant lineage: FP8/FP4 (dynamic fp8 weights, fp4 experts,
  ue8m0 block scales) → re-quantized to EXL3 by LibertAI.

### Architecture deltas vs anything PonyExl3 runs today
| Component | Spec | Notes for the port |
|---|---|---|
| CED | 40 layers = 20 encoder + 20 decoder; decoder global KV projected from final encoder hidden states | 8B active prefill / 16B decode — the source of the speed story |
| CSA2 attention | every layer has a 128-tok sliding window; global compressed KV; static modes Full / Reindex / Reuse; `kv_source_layer_ids [2,8,14,20]`, `compress_ratios` per layer | needs SWA-bounded-replay + compressed-KV + reuse caches in `mlx` cache classes |
| Indexer | 2-level DSA-style: `index_n_heads 32`, `index_head_dim 128`, `index_topk 512`, source layers `[2,8,14,20,24,28,32,36]` | top-k selection kernel, indexer KV cache |
| MLA variant | `q_lora_rank 1280`, `o_lora_rank 1024`, `o_groups 8`, `head_dim 512`, `qk_rope_head_dim 64`, `num_key_value_heads 1` | projection shapes differ from deepseek_v3; many are EXL3 linears |
| MoE | 384 routed + 1 shared, top-6, `scoring_func sqrtsoftplus`, `topk_method noaux_tc`, `routed_scaling_factor 1.5`, `swiglu_limit 10` | scales `EXL3SwitchGLU` up 6× vs Qwen MoE |
| mHC | hyper-connections, `hc_mult 4`, Sinkhorn 20 iters | elementwise/small-matmul heavy; compile-friendly |
| Engram | layers 1 and 14; two tables (~384 M rows each, compressed vocab 99 092, n-gram ≤ 4, 8 heads × 256) | read-only gather tables → mmap / SSD-paging candidate, NOT EXL3 |
| YaRN 1M | `factor 16`, `original_max 65536` | YaRN applied via rope_parameters path |
| DSpark | 3 MTP layers, markov rank 256, block size 5, its own 128-expert top-3 MoE | bundled at `mtp_bits: 4`; Phase 6b |
| Vision | 32-layer DeepSeek-ViT + projector | out of scope v1 |

### Reference implementations we can lean on
- **`PipeNetwork/deepseek-v41-mlx` (MIT)** — working Apple-Silicon MLX text
  implementation: `config/attention/sparse_attention/compressor/indexer/
  hyper_connections/engram/moe/cache/layers/model/load/generate` +
  `docs/reference/` (PyTorch reference from DeepSeek: `model.py`, `engram.py`,
  `encoding.py`, `kernel.py`) + `tests/test_parity.py`. This is the primary
  port source and parity oracle.
- vLLM 0.31+ / SGLang server implementations; TensorFold (CUDA) for the
  BlackwellBoy side.
- mlx-community / gbx-lm / oQ quants of the same checkpoint for cross-checks.

### PonyExl3 pieces we reuse as-is
- `_ARCHITECTURES` map + `mlx_lm`-skeleton loader (`mlx/model.py`), including
  the language_model-wrapper and unmapped-module skip (MTP head tolerant path).
- Per-module-K `EXL3Linear` build; `EXL3SwitchGLU` fused MoE stacking (Qwen
  35B-A3B path).
- `mul1` codebook + CPU `ref/` codec parity suite (293 tests).
- Generation loop with last-position `lm_head` (`mlx/generate.py`), MTP
  auto-discovery (`--mtp`), speculative-verify framework.
- Loader's incremental free + wired-cap pattern (`PONYEXL3_MEM_LIMIT_GB`).

---

## 2. Phases

### Phase 0 — green baseline + modern-container gate (D0–D1)
- [ ] `uv venv --python 3.14` (system py3.15 may lack mlx wheels), `pip install -e ".[dev]"`.
- [ ] `pytest tests/ -q` → 293 green on M3 Ultra (records our baseline vs upstream `v0.3.0`).
- [ ] Download `turboderp/Qwen3.8-27B-exl3` rev `4.00bpw` (~15 GB; container **v1.4.2**, `mul1`, `mtp_bits: 4`).
- [ ] `ponyexl3-generate … --mtp auto` smoke; then `--mtp off` parity check vs plain greedy.
- **GO gate:** container ≥ v1.4 loads and generates correctly → the v1.6.2 DS pack has
  a realistic chance. NO-GO → the loader-vs-1.4.x diff becomes the first project.
- Log: `docs/dsv41_devlog.md` starts here (every phase appends).

### Phase 1 — container + layout audit of the DS pack (D1–D3)
- [ ] Download LibertAI pack (~205 GB) + `engram-src` (DeepSeek Engram shards, ~189 GB).
- [ ] Tool `tools/dev/dsv41_layout.py`: dump `quantization_config.json` storage schema
      + `exl3_k_map.json`; diff module keys against what `_module_path()` /
      `_is_exl3_linear_key()`-style paths would expect (`model.layers.N.…` vs Qwen's
      `model.language_model.layers.N.…`; expert naming `…experts.<i>.…`;
      indexer/compressor/mHC keys; `mtp.*`; `visual.*`).
- [ ] Verify tile constraints on every quantized module: in/out multiples of 16
      (and HAD_DIM 128 where Hadamard applies) — flag odd dims early
      (`q_lora_rank 1280`, `o_lora_rank 1024`, indexer dims, `moe_intermediate 2304`, etc.).
- [ ] Audit Engram dir format: safetensors layout, dtypes (FP8), hash/n-gram encoding
      (cross-read PipeNetwork `docs/reference/engram.py` + `encoding/` in the pack).
- [ ] Identify every non-EXL3 tensor the bundle carries plain (norms, routers,
      `A_log`-style, dt_bias, embeddings, hc params).
- **Deliverables:** `docs/dsv41_layout_audit.md` (key inventory + gaps list +
  memory math) and an env-gated `tests/test_dsv41_bundle_layout.py` that
  asserts the audit facts (runs only with `PONYEXL3_MODEL_DSV41` set).

### Phase 2 — pure-MLX `deepseek_v41` text architecture (D3–D10)
No EXL3 yet; fp16/fakequant, so everything is testable without the 400 GB on disk.
- [ ] New package `ponyexl3/arch/deepseek_v41/` (name avoids mlx-lm dependency since
      `mlx_lm.models.deepseek_v41` does not exist): `config.py`, `cache.py`,
      `compressor.py`, `indexer.py`, `attention.py` (CSA2: SWA + global compressed KV,
      Full/Reindex/Reuse), `moe.py` (sqrtsoftplus/noaux_tc router, swiglu_limit),
      `hyper_connections.py` (mHC Sinkhorn), `engram.py` (n-gram hash lookup + gather),
      `model.py` (CED wiring: encoder/decoder KV projection), `load.py` (HF FP8→dequant
      path optional; plain-tensor load for our bundles).
- [ ] Cache classes: SWA bounded-replay, compressed global KV (YaRN), indexer cache,
      Reuse-mode sharing.
- [ ] Parity tests vs PipeNetwork's implementation on a tiny random-config model
      (random `deepseek_v41` fixture generated by our own init script): token/logit
      equality within fp16 tolerance, per-submodule unit tests mirroring their
      `test_parity.py` cases + their `docs/reference` torch outputs as golden npz.
- [ ] Generation loop integration: cache building in `mlx/generate.py`
      (`make_cache()` equivalent), encoder/decoder phase split for prefill.
- **Exit:** synthetic tiny model runs greedy in our loop; submodule parity tests green.

### Phase 3 — EXL3 integration (D10–D16)
- [ ] Loader: extend `_ARCHITECTURES` with `deepseek_v41` / `deepseek_v41_text`
      resolving to our local arch module (add a non-`mlx_lm` import path in
      `mlx/model.py`).
- [ ] Tolerate container v1.6.2 fields (`avg_bits`, `mtp_bits`, version string).
- [ ] Map the pack's storage keys to our model tree (from Phase 1 audit); swap
      attention/indexer/compressor/shared-expert linears for `EXL3Linear`;
      build 384-expert `EXL3SwitchGLU` stacks × 40 layers (chunked build so the
      stack loop stays within load-transient memory; validate uint16-packed
      trellis concat sizes for 384-deep stacks).
- [ ] Router/gate + norms + hc + embed = plain tensors (already the loader's path).
- [ ] Skip `mtp.*` + `visual.*` on base load (existing unmapped-skip + sanitize drop).
- [ ] Engram: load FP8 tables via `np.memmap` → device upload or on-demand gather;
      Metal gather kernel if `mx.take` along hashed rows is too slow at 189 GB.
- **Exit:** full pack loads; `--engine fold16` vs `exl3` agreement report
  (`ponyexl3-compare-engines`) on short prompts; strict-load parity vs the
  Phase-2 fp16 path on the same real weights (fold16 as bridge).

### Phase 4 — memory engineering for 512 GB (D16–D20)
- [ ] Document/raise wired limit (`sudo sysctl iogpu.wired_limit_mb=…`); set a
      sane `PONYEXL3_MEM_LIMIT_GB` default for 512 GB boxes.
- [ ] Load-transient: per-layer free already exists — extend to expert stacks and
      Engram mmap (never materialize host fp16 copies of tables).
- [ ] Engram residency policy: full-resident (fast) vs mmap-lazy (default) vs
      explicit offload; measure tok/s delta; record footprint table in README.
- [ ] Stretch for ecosystem: SSD-paged experts path so 96–128 GB Macs can run a
      2.9 bpw pack (GreenBitAI paged precedent) — separate branch `feat/dsv41-paging`.

### Phase 5 — correctness + quality gates (D20–D24)
- [ ] Token-level: greedy outputs stable across reruns; `--mtp auto`/`--lookup`
  verify-gated token-identity holds on this arch.
- [ ] Cross-engine: our `exl3` vs PipeNetwork mixed-4/8 MLX checkpoint on shared
  prompts (same tokenizer): expected small drift (different quants), no qualitative break.
- [ ] Quality: mlx-eval KLD harness vs `Qwen/Qwen…`-style bf16 bridge →
  here: vs DeepSeek FP8 release dequantized (or the official API with recommended
  sampling as reference); ΔPPL / ΔAcc@1 table in the devlog.
- [ ] Add non-model CI tests for: router math (sqrtsoftplus/noaux_tc), Sinkhorn,
  engram hashing (vs golden vectors), cache class behavior.

### Phase 6 — performance (D24–D30)
- [ ] Kernel routing audit for this arch: fused GEMV at M=1 for MLA/mHC/indexer
      projections; batched top-6-of-384 expert gather inside the SwitchGLU kernels;
      indexer top-512 kernel; Sinkhorn 20-iter unroll under `mx.compile`.
- [ ] `ponyexl3-generate-bench` sweeps 1k–128k; targets (M3 Ultra, 2.9 bpw, 8k ctx):
      plain decode **≥ 15 tok/s** (stretch 25+), cold prefill ≥ 150 tok/s at 32k.
      These numbers are honest-first: publish whatever we measure.
- [ ] 6b (stretch): DSpark drafter port from config params (`dspark_*`, `mtp_bits 4`
      tensors in-pack) into the existing speculative framework; MTP-style verify
      keeps output token-identical.

### Phase 7 — ecosystem + upstream (D30+)
- [ ] `ponyexl3-serve`: OpenAI-compatible server over the generate loop
      (chat + streaming + tool-call passthrough; DSML template quirks documented).
- [ ] `ponyexl3-patch`: apply safetensors overlays (BlackwellBoy `wo_b` patch,
      heretic-style edits) to any EXL3 bundle on Mac — validates every output shard.
- [ ] Docs: README `deepseek_v41` rows (supported models, memory table, bench),
      devlog → results page; example `blackfold.env`-style recipes.
- [ ] Upstream PRs: `beamivalice/PonyExl3` (container 1.6 + arch port as requested),
      `ml-explore/mlx-lm#1233` synergy (our arch module is reusable there),
      notes back to BlackwellBoy/LibertAI/PipeNetwork threads.

---

## 3. Risks & mitigations

| Risk | Sev | Mitigation |
|---|---|---|
| Pack tensor layout deviates from what PonyExl3's per-module loader expects ("Mia-AiLab layout") | High | Phase 1 audit gates all coding; only Phase 3 mapping depends on it |
| Container v1.6.2 format drift vs v0.0.30/1.4.x | High | Phase 0 Qwen3.8 (v1.4.2) test is the canary; ref codec has version-agnostic fixtures |
| Engram semantics (hash encoding, compressed vocab) misport → silent quality loss | Med | golden-vector tests vs PipeNetwork `engram.py` + DeepSeek torch reference; ablation (engram off) sanity |
| Kernel perf on exotic shapes (head_dim 512, kv_heads 1, 384-deep stacks) | Med | Phase 6 audit; fallback to non-fused matmul path per-module (fold engines exist) |
| Load transient > 512 GB during expert stacking | Med | chunked stack build + incremental frees; measure in Phase 3/4 |
| BlackwellBoy gated access stalls | Low | entire plan runs on ungated LibertAI; BLACKFOLD becomes an overlay test later |
| mlx wheels for Python 3.15 missing | Low | pin venv to 3.14 (upstream-tested version) |
| 1M-context KV cache blowup in first cuts | Low | cap context in benches; FP4 KV (model trains with it) is a follow-up |

## 4. Downloads ledger (disk: keep ~1 TB free)
| Artifact | Size | When |
|---|---|---|
| turboderp Qwen3.8-27B-exl3 `4.00bpw` | ~15 GB | Phase 0 |
| LibertAIDAI DS-V4.1-Flash EXL3 2.9bpw | ~205 GB | Phase 1 |
| DeepSeek Engram shards (engram-src) | ~189 GB | Phase 1 |
| PipeNetwork DS-V4.1-Flash MLX mixed 4/8 (optional cross-check) | ~428 GB | Phase 5, optional |
| BlackwellBoy BLACKFOLD repo + setup outputs | ~1 GB patch | when access granted |

## 5. Working agreements
- Branch per phase: `feat/dsv41-p1-layout`, `feat/dsv41-p2-arch`, … → PR to
  `origin/master`; squash-merge; keep upstream sync via `upstream/master` merges.
- `pytest tests/ -q` green before every push; new tests never require a
  model on disk unless env-gated (`PONYEXL3_MODEL_DSV41`, `PONYEXL3_ENGRAM_DIR`).
- All measurements go in `docs/dsv41_devlog.md` with machine/commit/context tags.
- Honesty rule: numbers we publish are numbers we reproduced on the M3 Ultra.

## 6. Definition of done (v1)
`turboderp`-style one-liner: `ponyexl3-generate /path/to/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw --mtp auto`
loads the full pack on a 512 GB Apple Silicon Mac, produces coherent chat output
with the model's chat template, beats the Phase-6 decode target, ships with a
parity test suite, and the README documents it.
