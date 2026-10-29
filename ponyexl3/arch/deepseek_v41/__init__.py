"""``deepseek_v41`` (DeepSeek-V4.1-Flash) text architecture, pure MLX.

Ported from `PipeNetwork/deepseek-v41-mlx <https://github.com/PipeNetwork/deepseek-v41-mlx>`_
(MIT), which is itself a port of DeepSeek's torch reference (``docs/reference``).
Module attribute paths are kept **identical** to the checkpoint's inference-style
tensor names (``layers.N.attn.wq_a``, ``layers.N.ffn.experts.N.w1``, ``embed``,
``head`` …) so the Phase 3 EXL3 loader can swap quantized linears 1:1.

This package has no ``mlx_lm`` architecture dependency (mlx-lm has no
``deepseek_v41`` module yet — ml-explore/mlx-lm#1233).
"""

from ponyexl3.arch.deepseek_v41.config import ModelArgs
from ponyexl3.arch.deepseek_v41.model import Model

__all__ = ["ModelArgs", "Model"]
