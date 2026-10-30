"""Where does a dsv41 decode step actually go? Class-level timing fences.

Loads the real pack (engram ON unless DSV41_SKIP_ENGRAM=1), prefills once to
compile kernels, times N plain decode steps, then re-runs with per-class
mx.eval fences over Attention / MoE / Indexer / Compressor / Engram / ...
and prints the breakdown. Run under tools/dev/sushi_yield.sh."""

import argparse
import collections
import sys
import time
import traceback


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("-n", "--tokens", type=int, default=8)
    a = ap.parse_args()

    import mlx.core as mx
    from mlx_lm.utils import load_tokenizer

    from ponyexl3.mlx.model import load_model

    model, cfg = load_model(a.model, engine="exl3", verbose=True)
    inner = model.model  # _InnerModel view
    tok = load_tokenizer(a.model)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "What is the capital of France?"}],
        add_generation_prompt=True)

    cache = model.make_cache()
    ids = mx.array([prompt])

    # prefill (compiles kernels — excluded from timing)
    t0 = time.perf_counter()
    h = inner(ids, cache=cache)
    logits = model.lm_head(h[:, -1:, :])
    mx.eval(logits)
    print(f"[prof] prefill {time.perf_counter() - t0:.2f}s", flush=True)

    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    t0 = time.perf_counter()
    y = mx.argmax(model.lm_head(inner(y[:, None], cache=cache)), axis=-1)
    mx.eval(y)
    print(f"[prof] first fenced decode step {time.perf_counter() - t0:.2f}s",
          flush=True)

    from ponyexl3.arch.deepseek_v41 import attention, compressor, engram, indexer, layers, moe
    from ponyexl3.mlx.exl3_linear import EXL3Linear
    from ponyexl3.mlx.exl3_moe import EXL3SwitchGLU
    targets = {
        attention.Attention: "Attention",
        attention.WoAGrouped: "WoAGrouped",
        moe.MoE: "MoE(all)",
        moe.SharedExpert: "SharedExpert",
        moe.Gate: "Gate",
        indexer.Indexer: "Indexer",
        compressor.Compressor: "Compressor",
        engram.Engram: "Engram(all)",
        engram.EngramEmbedding: "EngramEmbedding",
        layers.RMSNorm: "RMSNorm",
        EXL3SwitchGLU: "EXL3SwitchGLU",
        EXL3Linear: "EXL3Linear",
    }
    totals = collections.defaultdict(float)
    counts = collections.Counter()

    def instrument():
        for cls, label in targets.items():
            orig = cls.__call__

            def wrap(orig=orig, label=label):
                def f(self, *args, **kw):
                    t = time.perf_counter()
                    r = orig(self, *args, **kw)
                    try:
                        mx.eval(r)
                    except Exception:
                        pass
                    totals[label] += time.perf_counter() - t
                    counts[label] += 1
                    return r
                return f
            cls.__call__ = wrap(orig)
        # SwitchGLU internal path counters: which branch runs per step?
        for mname in ("_decode_fused", "_decode", "_prefill", "_gateup_fused",
                      "_down_mapped", "_call_3d"):
            orig = getattr(EXL3SwitchGLU, mname)

            def wrapm(orig=orig, mname=mname):
                def f(self, *args, **kw):
                    if mname == "_call_3d" and counts["__shapes__"] < 2:
                        x, idx = args
                        print(f"[prof] SwitchGLU x {tuple(x.shape)} {x.dtype}"
                              f" idx {tuple(idx.shape)} {idx.dtype}", flush=True)
                        counts["__shapes__"] += 1
                    t = time.perf_counter()
                    r = orig(self, *args, **kw)
                    mx.eval(r)
                    totals[f"SG.{mname}"] += time.perf_counter() - t
                    counts[f"SG.{mname}"] += 1
                    return r
                return f
            setattr(EXL3SwitchGLU, mname, wrapm(orig))
    instrument()

    # fresh cache; re-run prefill + decode with fences
    cache = model.make_cache()
    t0 = time.perf_counter()
    h = inner(ids, cache=cache)
    logits = model.lm_head(h[:, -1:, :])
    mx.eval(logits)
    print(f"[prof] fenced prefill {time.perf_counter() - t0:.2f}s", flush=True)
    prefill_totals = dict(totals)
    y = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(y)
    totals.clear(); counts.clear()

    t0 = time.perf_counter()
    for _ in range(a.tokens):
        y = mx.argmax(
            model.lm_head(inner(y[:, None], cache=cache))[:, -1, :], axis=-1)
        mx.eval(y)
    step_s = (time.perf_counter() - t0) / a.tokens
    print(f"[prof] fenced decode {step_s:.3f}s/step over {a.tokens} steps", flush=True)

    other = step_s - sum(totals.values()) / a.tokens
    print(f"{'component':18s} {'ms/step':>9s} {'calls':>6s}")
    for label, t in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"{label:18s} {1000 * t / a.tokens:9.1f} {counts[label] // a.tokens:6d}")
    print(f"{'(other)':18s} {1000 * other:9.1f}")
    print(f"[mem] device active {mx.get_active_memory()/2**30:.1f} GB "
          f"peak {mx.get_peak_memory()/2**30:.1f} GB")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
