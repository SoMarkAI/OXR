# MLX backend (Apple Silicon)

`oxr_model.backend: mlx` runs OXR-1.0 recognition **in-process** on Apple Silicon
through [MLX](https://github.com/ml-explore/mlx). Layout analysis stays the
separate service in `services/layout_analyze`; only the recognition model moves
in-process, so `oxr serve`, the demo and the HTTP API are unchanged.

This page collects the porting notes, the numerical validation and the measured
throughput that do not belong in the quick-start section of the README.

## Install

```bash
pip install -e ".[mlx]"
hf download SoMarkAI/OXR-1.0 --local-dir ./models/OXR-1.0
```

The extra adds `mlx`, `transformers` (tokenizer only), `pillow` and `numpy`.
Weights are read from the original checkpoint — there is no conversion step and no
second copy on disk.

## Configure

`config.yaml`:

```yaml
oxr_model:
  backend: mlx
  model_path: ./models/OXR-1.0
  dtype: bfloat16        # bfloat16 | float16 | float32
```

or the equivalent flags:

```bash
oxr serve --config config.yaml --workers 1 \
  --oxr-model-backend mlx --oxr-model-path ./models/OXR-1.0
```

`oxr_model.url` is unused when the backend is `mlx`. The model is loaded during
server startup (`model_client.warmup()`), so the first parse request does not pay
the weight-loading cost.

## Where the code lives

| File | Role |
|-|-|
| `oxr/model/base.py` | `OXRModelBackend` protocol and `ModelOutput` |
| `oxr/model/__init__.py` | `build_model()`, selects a backend from `oxr_model.backend` |
| `oxr/model/client.py` | `remote` backend (OpenAI-compatible HTTP client) |
| `oxr/model/local_mlx.py` | `mlx` backend: single worker thread, prompt routing |
| `oxr/model/mlx/model.py` | SigLIP2-NaViT tower, 65532-way visual vocabulary head, Qwen3-0.6B decoder |
| `oxr/model/mlx/preprocess.py` | `smart_resize`, bicubic resize, 2×2-block patchify |
| `oxr/model/mlx/runner.py` | prompt assembly, greedy decode with repetition penalty |
| `tests/mlx/` | manual parity and preprocessing checks (require torch + weights) |

## Validation

Checked against an independent float32 PyTorch implementation of the published math
(`tests/mlx/ref_torch.py`), on a document crop of 3432 patches (858 visual atoms):

| Stage | Max abs difference |
|-|-|
| patch vectors vs `transformers` | 0 (identical) |
| positional embedding interpolation | 1.4e-5 |
| vision features | 1.0e-2 (relative 5.5e-3) |
| visual soft tokens (65536-dim) | 1.0e-5 |
| prompt logits (876 tokens) | 5.9e-4, same argmax |

End-to-end, the markdown generated for a test page is identical to the hosted
API's output for the same file: heading structure, reading order, table contents
and `$E=mc^{2}$`.

```bash
python -m tests.mlx.parity path/to/crop.png      # needs weights + torch
python -m tests.mlx.check_preprocess path/to.png # preprocessing only
```

## Throughput

Reference measurement on an M-series laptop (64 GB), bf16, layout analysis on CPU:

| Workload | Time |
|-|-|
| model load during startup | ~5 s |
| one 1025×585 crop, ~60 output tokens | ~1.6 s |
| one-page PDF, layout + 6 blocks | ~4–11 s |
| three-page dense text PDF | ~52 s |

Memory: 3.07 GB of weights, ~4.1 GB resident.

## Porting notes

- `fullatt_block_indexes: null` means **full** attention in every vision layer. The
  window permutation is still applied, but it is a no-op under full attention.
- Indicator tokens (-301…-304) map to the **tail** of the visual vocabulary
  (65532…65535). Negative Python indices silently select the wrong rows.
- The visual head's `LayerNorm` uses torch's default `eps=1e-5`, not the config's
  `1e-6`.
- `mx.fast.rope` does not follow the `(batch, seq, heads, dim)` axis convention used
  by `transformers`, so RoPE is applied explicitly.
- MLX arrays belong to the thread (and stream) that created them, weights included.
  `local_mlx.py` therefore loads and runs the model on one dedicated worker thread.
- The pipeline renders PDFs at 150 DPI; small glyphs read poorly, which the model's
  own upscaling of crops to ≥448² mitigates but does not remove.

## Limitations

- bf16 by default; no quantisation yet.
- Requests are serialised: one MLX stream, one worker thread. This suits interactive
  and single-document use, not high-concurrency serving.
- Layout analysis runs on CPU on macOS (MPS is not wired up) and dominates dense
  pages.
