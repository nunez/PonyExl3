"""Load a ``deepseek_v41`` EXL3 bundle into ``ponyexl3.arch.deepseek_v41``.

Differs from the mlx_lm-skeleton path in ``model.py`` three ways:

* the checkpoint uses inference-style keys (``layers.N.attn.wq_a``,
  ``ffn.experts.N.w1``, ``embed``/``head``) that are *already* our arch's
  attribute paths — the remap is the identity, plus ``wo_a`` arriving as 8
  per-o-group slices assembled into ``WoAGrouped``;
* tensors are pulled through a header-indexed ``BundleReader`` (numpy, on
  demand) instead of ``mx.load``-ing every shard: the pack's plain ``mtp.*``
  / ``vision.*`` tensors carry F8 dtypes mlx's safetensors reader refuses,
  and we never want to materialize them text-only;
* the 384 routed experts per layer stack into EXL3SwitchGLU with the
  DeepSeek clamped SwiGLU (limit from config), K per layer from the trellis
  (K3, K2 on layers 18–22 — uniformity asserted, not assumed).

Engram embed tables are external (release FP8 shards): mmap'd uint8,
uploaded verbatim, dequantized per lookup. ``engram_dir`` resolution:
argument → config ``engram_table_dir`` → ``PONYEXL3_ENGRAM_DIR`` → skipped
with a warning (quality-ablation mode, tables stay zeroed).
"""

from __future__ import annotations

import gc
import json
import os
import re
import struct
import sys
from glob import glob
from pathlib import Path
from typing import Any

import mlx.core as mx
import ml_dtypes
import numpy as np
from mlx.utils import tree_flatten
from safetensors import safe_open

# safetensors' numpy framework resolves exotic dtypes via getattr(np, name);
# numpy 2.x does not expose ml_dtypes' float8 types that way. Register them
# so the F8_E4M3/F8_E8M0 engram/mtp tensors are readable (we view them as
# raw uint8 anyway).
for _n in ("float8_e8m0fnu", "float8_e4m3fn", "float8_e5m2", "float8_e4m3b11fnuz",
           "float8_e4m3fnuz", "float8_e5m2fnuz", "bfloat16", "int2", "int3", "int4"):
    if not hasattr(np, _n) and hasattr(ml_dtypes, _n):
        setattr(np, _n, getattr(ml_dtypes, _n))
del _n

from ponyexl3.arch.deepseek_v41 import WoAGrouped
from ponyexl3.arch.deepseek_v41.config import ModelArgs
from ponyexl3.arch.deepseek_v41.engram import build_compressed_token_map
from ponyexl3.arch.deepseek_v41.model import Model
from ponyexl3.arch.deepseek_v41.pony import DeepseekV41ForPony
from ponyexl3.mlx.exl3_linear import EXL3Linear
from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU
from ponyexl3.mlx.model import (
    _apply_device_memory_limit,
    _EXL3_SUFFIXES,
    _set_module,
    convert_engine,
    exl3_linears,
)
from ponyexl3.ref.codebook import MCG_MULT, MUL1_MULT, codebook_mode_from_flags
from ponyexl3.ref.layer import EXL3Layer

_EXPERT_RE = re.compile(r"layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])$")
_WO_SLICE_RE = re.compile(r"layers\.(\d+)\.attn\.wo_a\.slice\.(\d+)$")

_SKIP_PREFIXES = ("mtp.", "vision.", "aligner.", "image_start", "image_end", "image_newline")


def _np_to_mx(a: np.ndarray) -> mx.array:
    name = a.dtype.name
    if name == "bfloat16":
        return mx.array(np.ascontiguousarray(a).view(np.uint8)).view(mx.bfloat16)
    if name.startswith("float8"):
        return mx.array(np.ascontiguousarray(a).view(np.uint8))
    if name == "float8_e8m0fnu":  # view() above already catches it; belt
        return mx.array(np.ascontiguousarray(a).view(np.uint8))
    return mx.array(np.ascontiguousarray(a))


class BundleReader:
    """Header-indexed tensor access across a bundle's shards (numpy, lazy).

    Shards are read through ``safe_open`` mmaps; once every tensor of a
    shard has been consumed the handle is closed so its mapped pages can
    stop counting against the process footprint (a 196 GB pack otherwise
    keeps all 43 shards mapped for the whole load)."""

    def __init__(self, model_dir: str):
        self.shards = sorted(glob(os.path.join(model_dir, "*.safetensors")))
        if not self.shards:
            raise FileNotFoundError(f"no safetensors under {model_dir}")
        self._files: dict[str, Any] = {}
        self.keymap: dict[str, str] = {}
        self._pending: dict[str, int] = {}
        for p in self.shards:
            with safe_open(p, framework="numpy") as f:
                keys = list(f.keys())
                for k in keys:
                    self.keymap[k] = p
            self._pending[p] = len(keys)

    def has(self, key: str) -> bool:
        return key in self.keymap

    def numpy(self, key: str) -> np.ndarray:
        path = self.keymap.pop(key)
        fh = self._files.get(path)
        if fh is None:
            fh = self._files[path] = safe_open(path, framework="numpy")
        a = fh.get_tensor(key)  # copied bytes, not an mmap view
        self._pending[path] -= 1
        if self._pending[path] == 0:
            del self._files[path]  # last tensor read: drop the handle, unmap
        return a

    def close(self) -> None:
        self._files.clear()  # safetensors handles unmap when freed


def _layer_from_np(key: str, info: dict[str, Any], rd: BundleReader) -> EXL3Layer:
    trellis = rd.numpy(f"{key}.trellis").astype(np.uint16, copy=False)
    in_tiles, out_tiles, packed = trellis.shape
    k = packed * 16 // 256
    mcg = bool(info.get("mcg_multiplier"))
    mul1 = bool(info.get("mul1_multiplier"))
    if mcg and int(rd.numpy(f"{key}.mcg").astype(np.int64)) & 0xFFFFFFFF != int(MCG_MULT):
        raise ValueError(f"{key}: unsupported mcg multiplier")
    if mul1 and int(rd.numpy(f"{key}.mul1").astype(np.int64)) & 0xFFFFFFFF != int(MUL1_MULT):
        raise ValueError(f"{key}: unsupported mul1 multiplier")

    def _np16(suffix: str) -> np.ndarray | None:
        for s in (suffix, suffix.rstrip("h")):
            if rd.has(f"{key}.{s}"):
                return rd.numpy(f"{key}.{s}").astype(np.float16)
        return None

    return EXL3Layer(
        key=key,
        in_features=in_tiles * 16,
        out_features=out_tiles * 16,
        k=k,
        trellis=trellis,
        suh=_np16("suh"),
        svh=_np16("svh"),
        bias=_np16("bias"),
        mcg=mcg,
        mul1=mul1,
    )


def _cb_for(info: dict[str, Any]):
    return codebook_mode_from_flags(
        mcg=bool(info.get("mcg_multiplier")), mul1=bool(info.get("mul1_multiplier"))
    )


def _build_expert_stacks(
    model: Model, storage: dict[str, dict[str, Any]], rd: BundleReader,
    swiglu_limit: float, verbose: bool,
) -> set[str]:
    """Per layer, stack w1/w3/w2 of all experts into one EXL3SwitchGLU."""
    layers: dict[int, int] = {}
    for key in storage:
        m = _EXPERT_RE.match(key)
        if m:
            li, e = int(m.group(1)), int(m.group(2))
            layers[li] = max(layers.get(li, 0), e + 1)
    if not layers:
        return set()

    consumed: set[str] = set()
    for li, E in sorted(layers.items()):
        pre = f"layers.{li}.ffn.experts"
        gu_t, gu_u, dn_t = [], [], []
        gu_suh, gu_svh, dn_suh, dn_svh = [], [], [], []
        k: int | None = None
        cb = None
        for e in range(E):
            lg = _layer_from_np(f"{pre}.{e}.w1", storage[f"{pre}.{e}.w1"], rd)
            lu = _layer_from_np(f"{pre}.{e}.w3", storage[f"{pre}.{e}.w3"], rd)
            ld = _layer_from_np(f"{pre}.{e}.w2", storage[f"{pre}.{e}.w2"], rd)
            consumed |= {f"{pre}.{e}.w1", f"{pre}.{e}.w2", f"{pre}.{e}.w3"}
            for layer in (lg, lu, ld):
                if k is None:
                    k, cb = layer.k, _cb_for(storage[f"{pre}.{e}.w1"])
                elif layer.k != k:
                    raise ValueError(
                        f"{pre}.{e}: K {layer.k} != stack K {k} — mixed-K "
                        "experts inside one layer are not supported")
            gu_t.append(lg.trellis)  # gate trellises first, then up — the
            gu_u.append(lu.trellis)  # EXL3SwitchGLU stacking convention
            gu_suh.append(np.stack([lg.suh, lu.suh]))
            gu_svh.append(np.concatenate([lg.svh, lu.svh]))
            dn_t.append(ld.trellis)
            dn_suh.append(ld.suh)
            dn_svh.append(ld.svh)
            del lg, lu, ld
        assert k is not None and cb is not None
        mod = EXL3SwitchGLU(
            gu_trellis=mx.array(np.concatenate(gu_t + gu_u, axis=1)),
            gu_suh=mx.array(np.stack(gu_suh).astype(np.float16)),
            gu_svh=mx.array(np.stack(gu_svh).astype(np.float16)),
            dn_trellis=mx.array(np.concatenate(dn_t, axis=1)),
            dn_suh=mx.array(np.stack(dn_suh).astype(np.float16)),
            dn_svh=mx.array(np.stack(dn_svh).astype(np.float16)),
            k=k, cb=cb, activation="silu", swiglu_limit=swiglu_limit,
        )
        del gu_t, gu_u, dn_t, gu_suh, gu_svh, dn_suh, dn_svh
        gc.collect()
        mx.eval(mod.parameters())
        _set_module(model, f"layers.{li}.ffn.experts", mod)
        mx.clear_cache()
        if verbose:
            print(f"  moe layer {li}: {E} experts, k={k}, clamp={swiglu_limit:g}")
    return consumed


def _mmap_u8(path: str, key: str) -> np.ndarray:
    """Byte-range mmap of a safetensors tensor as uint8 (no copy).

    The F8 tables are 94 GiB each; ``get_tensor`` would allocate that as
    dirty anonymous memory on top of the device upload. A read-only mmap
    view keeps the host pages file-backed (reclaimable under pressure)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
    meta = header[key]
    lo, hi = meta["data_offsets"]
    base = 8 + hlen
    rows = meta["shape"][0]
    cols = int(meta["shape"][1]) if len(meta["shape"]) > 1 else 1
    if (hi - lo) != rows * cols:
        raise ValueError(f"{key}: byte range != {rows}x{cols} uint8")
    return np.memmap(path, dtype=np.uint8, mode="c",
                     offset=base + lo, shape=(rows, cols))


def _load_engram_tables(
    model: Model, args: ModelArgs, engram_dir: str | None, verbose: bool,
) -> bool:
    """Upload external FP8 embed tables (uint8 bytes + ue8m0 scales)."""
    if not args.engram_layer_ids:
        return True
    if not engram_dir:
        print("[load] WARNING: no engram table dir — tables stay zeroed "
              "(engram contribution disabled). Pass --engram-dir or set "
              "PONYEXL3_ENGRAM_DIR.", file=sys.stderr)
        return False
    keymap: dict[str, tuple[str, str]] = {}
    for p in sorted(glob(os.path.join(engram_dir, "*.safetensors"))):
        with open(p, "rb") as f:
            (hlen,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(hlen).decode("utf-8"))
        for k, meta in header.items():
            if k != "__metadata__":
                keymap[k] = (p, meta["dtype"])
    ok = True
    for li in args.engram_layer_ids:
        embed = model.layers[li].engram.embed
        views = []
        for suffix in ("weight", "scale"):
            key = f"layers.{li}.engram.embed.{suffix}"
            if key not in keymap:
                print(f"[load] WARNING: {key} not found in {engram_dir}", file=sys.stderr)
                ok = False
                break
            path, dtype = keymap[key]
            if dtype not in ("F8_E4M3", "F8_E8M0", "U8"):
                raise ValueError(f"{key}: unexpected dtype {dtype} for embed table")
            view = _mmap_u8(path, key)
            views.append(view)
            if verbose:
                print(f"  engram layer {li}.{suffix}: {tuple(view.shape)} "
                      f"{view.nbytes / 2**30:.1f} GiB host-resident (mmap)")
        if len(views) == 2:
            embed.set_host_tables(*views)   # rows gathered on CPU per lookup
        else:
            ok = False
    return ok


def _ensure_token_map(model_dir: str, args: ModelArgs, verbose: bool) -> list[int] | None:
    """The compressed-token map (cached as dsv41_token_map.json next to the
    tokenizer); hard-errors if it disagrees with the config's declared size."""
    if not args.engram_layer_ids:
        return None
    cache = os.path.join(model_dir, "dsv41_token_map.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return json.load(f)
    tok_file = os.path.join(model_dir, "tokenizer.json")
    if not os.path.exists(tok_file):
        raise FileNotFoundError(
            f"{model_dir}: engram layers need tokenizer.json to build the "
            "compressed-token map")
    from transformers import PreTrainedTokenizerFast

    from time import perf_counter

    tic = perf_counter()
    tok = PreTrainedTokenizerFast(tokenizer_file=tok_file)
    lookup, n = build_compressed_token_map(tok)
    if args.engram_compressed_vocab_size and n != args.engram_compressed_vocab_size:
        raise ValueError(
            f"compressed vocab {n} != config {args.engram_compressed_vocab_size}; "
            "the hash multipliers derive from it — a mismatch silently rehashes "
            "the engram table")
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(lookup, f)
    if verbose:
        print(f"  token map built: {n} classes in {perf_counter() - tic:.1f}s")
    return lookup


def load_dsv41_model(
    model_dir: str,
    *,
    engine: str = "exl3",
    warm: bool = True,
    verbose: bool = False,
    group_size: int = 64,
    report_errors: bool = False,
    engram_dir: str | None = None,
    load_engram: bool = True,
) -> tuple[DeepseekV41ForPony, dict[str, Any]]:
    _apply_device_memory_limit()
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    with open(os.path.join(model_dir, "quantization_config.json"), encoding="utf-8") as f:
        qcfg = json.load(f)
    if qcfg.get("quant_method") != "exl3":
        raise ValueError(f"{model_dir} is not an EXL3 checkpoint")
    storage = {k: v for k, v in qcfg["tensor_storage"].items()
               if v.get("quant_format") == "exl3"}

    args = ModelArgs.from_dict(config)
    token_map = _ensure_token_map(model_dir, args, verbose)
    model = Model(args, token_map=token_map)

    rd = BundleReader(model_dir)

    # 1) routed experts → EXL3SwitchGLU stacks (clamped SwiGLU from config)
    consumed = _build_expert_stacks(model, storage, rd, args.swiglu_limit, verbose)

    # 2) wo_a per-o-group slices → WoAGrouped
    wo_slices: dict[int, dict[int, str]] = {}
    for key in storage:
        m = _WO_SLICE_RE.match(key)
        if m:
            wo_slices.setdefault(int(m.group(1)), {})[int(m.group(2))] = key
    for li, groups in sorted(wo_slices.items()):
        n = max(groups) + 1
        if sorted(groups) != list(range(n)):
            raise ValueError(f"layer {li}: wo_a slices not contiguous 0..{n - 1}")
        mods = [EXL3Linear(_layer_from_np(groups[g], storage[groups[g]], rd))
                for g in range(n)]
        _set_module(model, f"layers.{li}.attn.wo_a", WoAGrouped(mods))
        consumed |= set(groups.values())
        mx.clear_cache()
        if verbose:
            print(f"  wo_a layer {li}: {n} group slices")

    # 3) remaining EXL3 modules → EXL3Linear (identity key → attribute path)
    n_exl3 = 0
    for key, info in sorted(storage.items()):
        if key in consumed:
            continue
        _set_module(model, key, EXL3Linear(_layer_from_np(key, info, rd)))
        n_exl3 += 1
        mx.clear_cache()
    if verbose:
        print(f"  {n_exl3} other EXL3 modules swapped "
              f"({len(consumed)} consumed by stacks/slices)")

    # 4) plain tensors (never the skipped namespaces, never F8-parsing mx.load)
    exl3_subtensors = {f"{mod}.{suf}" for mod in storage for suf in _EXL3_SUFFIXES}
    loaded: set[str] = set()
    skipped_ns = 0
    plain_items: list[tuple[str, mx.array]] = []
    for key in sorted(rd.keymap):
        if any(key.startswith(p) for p in _SKIP_PREFIXES):
            skipped_ns += 1
            continue
        if key in exl3_subtensors:
            continue  # consumed module-wise above
        plain_items.append((key, _np_to_mx(rd.numpy(key))))
    model.load_weights(plain_items, strict=False)
    loaded = {k for k, _ in plain_items}
    expected = {k for k, _ in tree_flatten(model.parameters())}
    missing = expected - loaded
    extra = loaded - expected
    if extra:
        raise ValueError(f"{len(extra)} checkpoint tensors have no home, "
                         f"e.g. {sorted(extra)[:4]}")
    # engram embed tables arrive from the external FP8 dir, not the bundle
    engram_keys = {f"layers.{li}.engram.embed.{s}"
                   for li in args.engram_layer_ids for s in ("weight", "scale")}
    missing -= engram_keys
    table_ok = False
    if args.engram_layer_ids and load_engram \
            and os.environ.get("DSV41_SKIP_ENGRAM", "") != "1":
        table_ok = _load_engram_tables(model, args, engram_dir, verbose)
    elif args.engram_layer_ids:
        # ablation mode: zero-contribution tables, but don't leave the lazy
        # 384M-row placeholders for the final parameters() eval to inflate
        for li in args.engram_layer_ids:
            emb = model.layers[li].engram.embed
            emb.weight = mx.zeros((1, emb.weight.shape[1]), dtype=mx.uint8)
            emb.scale = mx.zeros((1, emb.scale.shape[1]), dtype=mx.uint8)
    if missing:
        raise ValueError(f"{len(missing)} params missing from checkpoint, "
                         f"e.g. {sorted(missing)[:6]}")
    rd.close()
    del plain_items
    mx.clear_cache()
    if verbose:
        print(f"  plain tensors: {len(loaded)} loaded, {skipped_ns} skipped "
              f"({', '.join(_SKIP_PREFIXES[:3])}…)")

    # 5) engine conversion (fold16/w8a16/… operate on the EXL3Linear slots)
    errors = convert_engine(
        model, engine, group_size=group_size, report_errors=report_errors,
        verbose=verbose,
        sidecar_dir=os.path.join(model_dir, ".pony_cache",
                                 f"gptq_w4g{group_size}") if engine == "w4gptq" else None,
    )
    if errors:
        worst = max(errors, key=lambda e: e.rel)
        print(f"[{engine}] requantized {len(errors)} layers — worst rel RMS "
              f"err {worst.rel:.2e} ({worst.key}). Lossy vs EXL3; validate.")

    if engine == "exl3":
        released = 0
        for _, m in exl3_linears(model):
            released += 1 if getattr(m._exl3, "trellis", None) is not None else 0
            m.release_source()
        mx.clear_cache()
        if verbose:
            print(f"  released host trellis for {released} layers")
    if warm and engine == "exl3":
        for _, m in exl3_linears(model):
            m.warm()
    mx.eval(model.parameters())
    mx.clear_cache()
    model.eval()
    if verbose and not table_ok and args.engram_layer_ids:
        print("[load] engram tables NOT loaded — outputs will differ from "
              "the reference model", file=sys.stderr)
    return DeepseekV41ForPony(model), config
