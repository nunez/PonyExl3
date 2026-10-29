# EXL3 layout audit: `DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw`

## Container
- config `model_type`: `deepseek_v41` | text_config: True | vision_config: True
- `quant_method`: 'exl3'
- `version`: '1.6.2'
- `bits`: 2.9
- `head_bits`: 6
- `out_scales`: 'always'
- `codebook`: 'mul1'
- `mtp_bits`: 4
- `exl3_k_map.json` present (external per-module K manifest)
- EXL3 modules (`tensor_storage`, `quant_format=exl3`): **46702**

## EXL3 modules by namespace
- `head`: 1
- `layers.*`: 46701

## Per-module K (derived from trellis tile shapes)
- K2 (qcfg `bits_per_weight`: 2): 5760 modules
- K3 (qcfg `bits_per_weight`: 3): 40320 modules
- K4 (qcfg `bits_per_weight`: 4): 57 modules
- K5 (qcfg `bits_per_weight`: 5): 558 modules
- K6 (qcfg `bits_per_weight`: 6): 3 modules
- K8 (qcfg `bits_per_weight`: 8): 4 modules

- derived-K vs qcfg bits_per_weight mismatches: **0**

- dims not %16 (trellis-invalid): **0**
- dims not %128 (Hadamard-block skip candidates): **0**

## Plain (non-EXL3) tensors by namespace  [shard-header scan]
- `aligner.*`: 4 tensors, 0.14 GiB
- `embed.*`: 1 tensors, 1.23 GiB
- `image_end`: 1 tensors, 0.00 GiB
- `image_newline`: 1 tensors, 0.00 GiB
- `image_start`: 1 tensors, 0.00 GiB
- `layers.*`: 580 tensors, 0.30 GiB
- `mtp.*`: 4839 tensors, 6.88 GiB
- `norm.*`: 1 tensors, 0.00 GiB
- `vision.*`: 259 tensors, 0.77 GiB
- total plain: **9.31 GiB**
