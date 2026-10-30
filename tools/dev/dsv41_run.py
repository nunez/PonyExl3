"""Real-pack dsv41 generate runner that keeps full tracebacks.

``ponyexl3-generate`` converts ValueError to SystemExit(str) and hides the
frames; this runner does the same call chain (load_model + mlx_lm tokenizer
+ generate_text) but lets exceptions print. Run under
``tools/dev/sushi_yield.sh`` on the full pack."""

import argparse
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("-p", "--prompt", default="Hello!")
    ap.add_argument("-n", "--max-tokens", type=int, default=32)
    ap.add_argument("--engine", default="exl3")
    ap.add_argument("--temp", type=float, default=0.0)
    a = ap.parse_args()

    from mlx_lm.utils import load_tokenizer

    from ponyexl3.mlx.generate import generate_text
    from ponyexl3.mlx.model import load_model

    model, cfg = load_model(a.model, engine=a.engine, verbose=True)
    tokenizer = load_tokenizer(a.model)
    try:
        text, stats = generate_text(
            model, tokenizer, a.prompt,
            max_tokens=a.max_tokens, temp=a.temp,
            on_segment=lambda s: print(s, end="", flush=True),
        )
    except Exception:
        traceback.print_exc()
        return 1
    print()
    print(f"[gen] {stats.summary()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
