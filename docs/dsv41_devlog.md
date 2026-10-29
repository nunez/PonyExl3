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

_(pending)_
