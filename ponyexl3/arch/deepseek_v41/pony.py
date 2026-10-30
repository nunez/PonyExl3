"""Adapter presenting ``Model`` with the mlx_lm-style tree ponyexl3.mlx.generate expects.

``generate._lm_parts`` splits a model into ``(lm, inner, lm_head)`` and drives
``lm.make_cache()`` / ``inner(toks, cache=...)`` / ``lm_head(hidden)``. The V4.1
Model keeps the release's flat checkpoint naming (``embed``/``layers.*``/
``norm``/``head``), so rather than renaming the tree we hand generate a thin
view: ``.model`` returns post-norm hiddens via ``Model.hidden``, ``.lm_head``
is ``Model.head``. Phase 3's EXL3 loader will build ``Model`` directly and
wrap it here, swapping ``head``/linears for EXL3 modules underneath.
"""

from __future__ import annotations

import mlx.core as mx

from ponyexl3.arch.deepseek_v41.model import Model


class _InnerModel:
    """Callable with the mlx_lm inner-model signature (mask kwargs ignored)."""

    def __init__(self, model: Model):
        self._m = model

    def __call__(self, toks: mx.array, cache=None, mask=None, **_kw) -> mx.array:
        return self._m.hidden(toks, cache)


class DeepseekV41ForPony:
    """``.model`` + ``.lm_head`` + ``make_cache()`` façade around Model."""

    def __init__(self, model: Model):
        self._m = model
        self.model = _InnerModel(model)
        self.lm_head = model.head
        # mlx_lm convention: generate/compare tools reach for
        # ``model.language_model`` before ``.model``/``.lm_head``
        self.language_model = self

    def make_cache(self, bsz: int = 1, max_seq_len: int | None = None,
                   dtype=mx.float32):
        return self._m.make_cache(bsz, max_seq_len, dtype)

    # --- delegate the usual mlx_lm model surface to the wrapped Model ---
    def parameters(self):
        return self._m.parameters()

    def named_modules(self):
        return self._m.named_modules()

    def eval(self):
        self._m.eval()
        return self

    @property
    def layers(self):
        return self._m.layers

    def __getattr__(self, name: str):
        if name == "_m":
            raise AttributeError(name)
        return getattr(self._m, name)
