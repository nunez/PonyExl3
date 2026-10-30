"""Is the real pack's EXL3SwitchGLU slow in isolation, or only in context?

Loads the pack (or attach-less quick exit), then times, for a couple of
layers, the synthetic-fast path against the real trellis arrays:
  - _gateup_fused(x2d, sel) xN
  - _down_mapped(h, sel) xN
  - __call__(x, idx) xN
plus the same with a freshly materialized x/sel. Under sushi_yield.sh."""

import argparse
import time
import traceback


def bench(label, fn, iters=10):
    fn(); mx_eval_flush()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    mx_eval_flush()
    print(f"  {label:34s} {(time.perf_counter() - t0) / iters * 1000:8.1f} ms",
          flush=True)


def mx_eval_flush():
    import mlx.core as mx
    mx.eval(mx.array(0))  # placeholder; real fence set below per call site


def main() -> int:
    global mx_eval_flush
    import mlx.core as mx
    import numpy as np

    def mx_eval_flush():
        mx.eval(_fence)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    a = ap.parse_args()
    from ponyexl3.mlx.model import load_model

    model, cfg = load_model(a.model, engine="exl3", verbose=False)
    m = model._m
    _fence = mx.array([0])
    mx.eval(_fence)
    rng = np.random.default_rng(11)
    for li in (5, 18):
        mod = m.layers[li].ffn.experts
        D, H = mod.input_dims, mod.hidden_dims
        x = mx.array(rng.random((1, D), dtype=np.float32).astype(np.float16))
        sel = mx.array(np.array([3, 111, 57, 290, 13, 377, 88, 200], np.int32))
        h = mx.array(rng.random((8, H), dtype=np.float32).astype(np.float16))
        mx.eval(x, sel, h)
        print(f"layer {li} (k={mod._k}, E={mod.num_experts}, D={D}, H={H}):",
              flush=True)
        bench("gateup_fused", lambda: mx.eval(mod._gateup_fused(x, sel)))
        bench("down_mapped", lambda: mx.eval(mod._down_mapped(h, sel)))
        x3 = mx.array(rng.random((1, 1, D), dtype=np.float32).astype(np.float16))
        idx = mx.array(np.array([[3, 111, 57, 290, 13, 377, 88, 200]],
                                np.int32)[None, :, :])
        mx.eval(x3, idx)
        bench("call (B=1,S=1)", lambda: mx.eval(mod(x3, idx)))
    print("[mem] device active %.1f GB" % (mx.get_active_memory() / 2**30))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
