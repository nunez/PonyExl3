"""Layout audit for an EXL3 bundle (dev tool, no model load).

Reads config.json + quantization_config.json (+ exl3_k_map.json when present)
and the safetensors shard *headers* only — never materializes weights — so it
runs in seconds on 400 GB-class packs. Produces a markdown report:

- EXL3 module inventory: per-module K (derived from trellis tile shapes),
  codebook flags, prefix classification, 16/128-divisibility flags
- plain (non-EXL3) tensor inventory grouped by top-level namespace
- unmapped-head inventory (mtp.*, visual.*) that the loader skips

Usage:
    python tools/dev/bundle_layout.py /path/to/exl3-bundle [--out report.md]

Designed for the deepseek_v41 effort (docs/deepseek_v41_plan.md Phase 1) but
architecture-agnostic; validated against Qwen3.8-27B-exl3 4.00bpw.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open

EXL3_SUFFIXES = ("trellis", "suh", "su", "svh", "sv", "mcg", "mul1", "bias")


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def namespace(key: str) -> str:
    parts = key.split(".")
    if "mtp" in parts:
        return "mtp.*"
    if "visual" in parts or parts[0] == "vision_tower":
        return "vision.*"
    if parts[0] == "model" and len(parts) > 1:
        return f"model.{parts[1]}.*"
    return parts[0] + (".*" if len(parts) > 1 else "")


def _trellis_shape(info: dict[str, Any]) -> list[int] | None:
    """qcfg `stored_tensors` entries are keyed by the *fully qualified*
    tensor name (`"<module>.trellis"`), not the bare suffix — mirror the
    loader's lookup in ponyexl3/mlx/model.py."""
    st = info.get("stored_tensors", {})
    for k, v in st.items():
        if k == "trellis" or k.endswith(".trellis"):
            shape = v.get("shape")
            if shape and len(shape) == 3:
                return list(shape)
    return None


def module_k(info: dict[str, Any]) -> int | None:
    shape = _trellis_shape(info)
    if shape is None:
        return None
    return int(shape[2]) * 16 // 256


def module_dims(info: dict[str, Any]) -> tuple[int, int]:
    """EXL3 module in/out dims. qcfg carries no input_features/output_features
    fields; trellis tiles are 16x16, so in = in_tiles*16, out = out_tiles*16."""
    shape = _trellis_shape(info)
    if shape is not None:
        return int(shape[0]) * 16, int(shape[1]) * 16
    return int(info.get("input_features", 0)), int(info.get("output_features", 0))


def audit(model_dir: Path) -> str:
    cfg_path = model_dir / "config.json"
    qcfg_path = model_dir / "quantization_config.json"
    if not cfg_path.is_file() or not qcfg_path.is_file():
        raise SystemExit(f"{model_dir}: need config.json + quantization_config.json")
    cfg = _read_json(cfg_path)
    qcfg = _read_json(qcfg_path)
    storage = {
        k: v
        for k, v in qcfg.get("tensor_storage", {}).items()
        if v.get("quant_format") == "exl3"
    }
    lines: list[str] = []
    add = lines.append

    add(f"# EXL3 layout audit: `{model_dir.name}`\n")
    add("## Container")
    qc = {k: v for k, v in qcfg.items() if not isinstance(v, dict)}
    add(f"- config `model_type`: `{cfg.get('model_type')}` | text_config: "
        f"{'text_config' in cfg} | vision_config: {'vision_config' in cfg}")
    for k, v in qc.items():
        add(f"- `{k}`: {v!r}")
    if (model_dir / "exl3_k_map.json").is_file():
        add("- `exl3_k_map.json` present (external per-module K manifest)")
    add(f"- EXL3 modules (`tensor_storage`, `quant_format=exl3`): **{len(storage)}**")

    # -- EXL3 module inventory -------------------------------------------------
    by_ns: dict[str, list[str]] = defaultdict(list)
    k_hist: dict[tuple[int, Any], int] = defaultdict(int)
    k_mismatch: list[str] = []
    odd16: list[str] = []
    odd128: list[str] = []
    for key, info in storage.items():
        by_ns[namespace(key)].append(key)
        k = module_k(info)
        bpw = info.get("bits_per_weight")
        if k is not None:
            if bpw is not None and int(bpw) != k:
                k_mismatch.append(f"{key} (derived K{k} vs qcfg bits {bpw})")
            k_hist[(k, bpw)] += 1
        ins, outs = module_dims(info)
        if ins % 16 or outs % 16:
            odd16.append(f"{key} ({ins}x{outs})")
        if ins % 128 or outs % 128:
            odd128.append(f"{key} ({ins}x{outs})")
    add("\n## EXL3 modules by namespace")
    for ns in sorted(by_ns):
        add(f"- `{ns}`: {len(by_ns[ns])}")
    add("\n## Per-module K (derived from trellis tile shapes)")
    for (k, bpw) in sorted(k_hist, key=lambda t: (t[0], str(t[1]))):
        add(f"- K{k} (qcfg `bits_per_weight`: {bpw}): {k_hist[(k, bpw)]} modules")
    add(f"\n- derived-K vs qcfg bits_per_weight mismatches: **{len(k_mismatch)}**")
    for m in k_mismatch[:20]:
        add(f"  - {m}")
    add(f"\n- dims not %16 (trellis-invalid): **{len(odd16)}**")
    for m in odd16[:20]:
        add(f"  - {m}")
    add(f"- dims not %128 (Hadamard-block skip candidates): **{len(odd128)}**")
    for m in odd128[:20]:
        add(f"  - {m}")
    if len(odd128) > 20:
        add(f"  - … {len(odd128) - 20} more")

    # -- shard header sweep: plain tensors -------------------------------------
    plain_ns: dict[str, tuple[int, int]] = {}  # ns -> (count, bytes)
    exl3_tensor_keys = {f"{m}.{s}" for m in storage for s in EXL3_SUFFIXES}
    shards = sorted(glob(str(model_dir / "*.safetensors")))
    if not shards:
        add("\n## Plain tensors\n- no safetensors shards found")
    else:
        for p in shards:
            with safe_open(p, framework="np") as f:
                for tk in f.keys():
                    if tk in exl3_tensor_keys:
                        continue
                    sl = f.get_slice(tk)
                    shape = sl.get_shape()
                    nbytes = int(np.prod(shape)) * {
                        "F32": 4, "F16": 2, "BF16": 2, "U8": 1, "I8": 1,
                        "U16": 2, "I16": 2, "I32": 4, "I64": 8, "F64": 8, "BOOL": 1,
                    }.get(str(sl.get_dtype()).split(".")[-1], 2)
                    ns = namespace(tk)
                    c, b = plain_ns.get(ns, (0, 0))
                    plain_ns[ns] = (c + 1, b + nbytes)
        add("\n## Plain (non-EXL3) tensors by namespace  [shard-header scan]")
        total = 0
        for ns in sorted(plain_ns):
            c, b = plain_ns[ns]
            total += b
            add(f"- `{ns}`: {c} tensors, {b / 2**30:.2f} GiB")
        add(f"- total plain: **{total / 2**30:.2f} GiB**")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    report = audit(args.model_dir)
    if args.out:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
