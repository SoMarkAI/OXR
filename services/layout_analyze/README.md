# D-FINE Layout Analysis Service

[![HuggingFace Model](https://img.shields.io/badge/HuggingFace-OXR--1.0-F59E0B?style=flat-square)](https://huggingface.co/SoMarkAI/OXR-1.0)
[![ModelScope Model](https://img.shields.io/badge/ModelScope-OXR--1.0-624AFF?style=flat-square)](https://modelscope.cn/models/SoMarkAI/OXR-1.0)

A small CPU/GPU HTTP service for a fixed D-FINE Stage-2 B2 layout-analysis
model. This directory contains inference code for OXR's all-in-one image. Model
weights are mounted at runtime and are not part of the source distribution.

## Supported model

- Architecture: HGNetv2 B2 + D-FINE Stage-2
- Input: RGB images, resized and letterboxed to `960 x 960`
- Classes: 12
- Decoder layers: 4
- Precision: FP16 autocast on CUDA; FP32 on CPU
- Fast path: Batch1 CUDA Graph

The service intentionally does not provide training, model registration,
authentication, a task queue, or a web UI.

## Requirements

- Linux
- For Docker deployment, use the [OXR all-in-one image](../../README.md#docker).
- For native development, use Python 3.12 with PyTorch `2.10.0+cu129`,
  matching the supported all-in-one model runtime.

CUDA mode additionally requires an NVIDIA GPU, a recent driver compatible with
CUDA 12.9, and NVIDIA Container Toolkit for Docker deployment.

## Model checkpoint

The layout checkpoint is bundled in the official `SoMarkAI/OXR-1.0` model
repositories on [Hugging Face](https://huggingface.co/SoMarkAI/OXR-1.0) and
[ModelScope](https://modelscope.cn/models/SoMarkAI/OXR-1.0):

```text
OXR-1.0/layout_analysis/last.pth
OXR-1.0/layout_analysis/metadata.json
```

Prepare the complete release using the [model download instructions](../../README.md#model-download).
If using Git, run `git lfs pull` first. The all-in-one container does not download weights automatically and
mounts the complete model repository read-only at `/models/oxr-model`.

The service reads `metadata.json` alongside the checkpoint (or the explicit
`MODEL_METADATA_PATH`). Schema version 1 records the model version, checkpoint
filename, byte size, SHA-256, architecture, preprocessing and class mapping.
The model repository owns these values; the service does not hardcode a
particular checkpoint's version or checksum.
`model_version` must be a public `major.minor.patch` release version.

For the current release:

```text
size: 78791307 bytes
sha256: dff78ccf18439955362ca44a69eb28fb0b40222a7d278860589682de0176dc17
```

Startup fails before model loading if metadata is missing/malformed, weight
size or SHA-256 differs, or architecture, input size, class IDs or preprocessing
are unsupported. The checkpoint filename must match `MODEL_PATH`. Metadata is
not a Transformers configuration and cannot request arbitrary architectures.
Device, effective precision and CUDA Graph remain runtime settings.
`/health` and `/model/info` report the public model version and configuration,
not local checkpoint or metadata mount paths.

## Start for native development

For the supported single-GPU Docker deployment, follow the
[OXR Docker instructions](../../README.md#docker). The model services remain
internal to that container; the HTTP API is the only published endpoint.

To run only this service for development, execute the following from this
directory using the Python/PyTorch environment described above. Keep LA
dependencies separate from the vLLM environment:

```bash
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt
MODEL_PATH=/absolute/path/to/OXR-1.0/layout_analysis/last.pth \
  PORT=8081 .venv/bin/python -m dfine_la
```

Wait until the service reports healthy:

```bash
curl http://127.0.0.1:8081/health
```

Stop this foreground process with Ctrl-C. Native mode binds all interfaces;
only use it on a trusted development network.

## Inference

Upload one image:

```bash
curl -X POST http://127.0.0.1:8081/inference \
  -F 'images=@page.png'
```

Upload a batch by repeating the same multipart field:

```bash
curl -X POST http://127.0.0.1:8081/inference \
  -F 'images=@page-1.png' \
  -F 'images=@page-2.png'
```

Use one confidence threshold for every class:

```bash
curl -X POST http://127.0.0.1:8081/inference \
  -F 'images=@page.png' \
  -F 'confidence=0.5'
```

Optionally override selected numeric class IDs with a JSON object. Classes not
listed in `class_confidences` continue to use the global `confidence` value:

| Class ID | Layout category |
| ---: | --- |
| `0` | Caption |
| `1` | Footnote |
| `2` | Formula |
| `3` | Text |
| `4` | Page-footer |
| `5` | Page-header |
| `6` | Picture |
| `7` | Title |
| `8` | Table |
| `9` | Code Block |
| `10` | Stamp |
| `11` | Chemical Structure |

```bash
curl -X POST http://127.0.0.1:8081/inference \
  -F 'images=@page.png' \
  -F 'confidence=0.5' \
  -F 'class_confidences={"4":0.7,"8":0.3}'
```

In this example, Page-footer uses `0.7`, Table uses `0.3`, and every other
category inherits the global `0.5` threshold.

Or use the included Python client:

```bash
python examples/infer.py page.png --confidence 0.5
```

The response shape is:

```json
{
  "status": "success",
  "params": {},
  "data": {
    "labels": [[0]],
    "boxes": [[[10.0, 20.0, 100.0, 200.0]]],
    "scores": [[0.99]]
  }
}
```

Without confidence fields, `labels`, `boxes`, and `scores` contain the model's
300 raw top-query outputs for each image, preserving the original API behavior.
`confidence` and every per-class override must be between `0` and `1`, and a
prediction is kept when `score >= threshold`. Filtering preserves model order
and keeps labels, boxes, and scores aligned. The service does not apply NMS.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/inference` | Multipart image inference |
| `GET` | `/health` | Readiness and runtime configuration |
| `GET` | `/model/info` | Model and device information |

Every successful inference response includes a `Server-Timing` header with
request-read, decode, preprocess, queue, and inference durations.

`POST /inference` accepts repeated `images` files plus optional multipart fields
`confidence` (one global threshold) and `class_confidences` (a JSON object whose
keys are numeric class IDs). If both are omitted, no filtering is applied and
the response `params` remains `{}`. When either field is supplied, `params`
reports the effective global threshold and normalized class overrides.

## Runtime configuration

| Variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | `/models/layout_analysis/last.pth` | Checkpoint path; explicitly set by the all-in-one supervisor |
| `MODEL_METADATA_PATH` | `metadata.json` alongside `MODEL_PATH` | Model metadata path |
| `PORT` | `8000` | Native HTTP port; explicitly set by the all-in-one supervisor |
| `DFINE_DEVICE` | `cuda` | Inference device: `cuda`, `cpu`, or `auto` |
| `DFINE_CUDA_GRAPH` | `1` | Enable the Batch1 CUDA Graph path |
| `DFINE_PARALLEL_PREPROCESS` | `1` | Run CPU decode/preprocess outside the GPU critical section |
| `LOG_LEVEL` | `info` | Uvicorn log level |

CUDA Graph is used for Batch1 CUDA requests. Other batch sizes and all CPU
requests use eager inference. `DFINE_CUDA_GRAPH=1` is safely ignored when the
resolved device is CPU. The runtime deliberately loads one model instance per
container.

For native CPU development, start the service with:

```bash
MODEL_PATH=/absolute/path/to/OXR-1.0/layout_analysis/last.pth \
  PORT=8081 DFINE_DEVICE=cpu DFINE_CUDA_GRAPH=0 .venv/bin/python -m dfine_la
```

CPU inference uses the same checkpoint and API contract, but is expected to be
substantially slower than CUDA inference at the fixed `960 x 960` input size.

## Development tests

The unit and API-contract tests do not require a GPU or checkpoint:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

Final release validation should build the OXR all-in-one image, start it with
the real checkpoint on an NVIDIA GPU, and compare its outputs with the known
reference service. Its model runtime remains fixed to vLLM `0.19.1`, PyTorch
`2.10.0+cu129` and Transformers `5.6.2`; there is no separate LA Docker image.

## Third-party code

The model implementation is an inference-only adaptation of D-FINE. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`third_party/D-FINE/LICENSE`](third_party/D-FINE/LICENSE).

The surrounding project code is licensed under [Apache 2.0](../../LICENSE).
See also [NOTICE](../../NOTICE). The model checkpoint has separate distribution
terms stated by the model repositories and is not included in this code distribution.
