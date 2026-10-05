# Layout Analysis

D-FINE Stage-2 B2 layout model with 12 classes and 960 × 960 image input.
OXR handles content recognition and reading order.

## Model files

Download the complete OXR-1.0 model repository from
[Hugging Face](https://huggingface.co/SoMarkAI/OXR-1.0) or
[ModelScope](https://modelscope.cn/models/SoMark/OXR-1.0). It includes:

- `layout_analysis/last.pth`: layout model weights.
- `layout_analysis/metadata.json`: model identity and checkpoint SHA-256.

Both files are required. Startup validates the metadata and weight integrity.

## Runtime

The supported LA runtime uses Python 3.12 and PyTorch `2.11.0+cu130`.
CUDA inference requires an NVIDIA GPU with a CUDA 13.0-compatible driver.
Inference uses FP16 on CUDA and FP32 on CPU.

## License

Project code uses the [OXR custom license](../../LICENSE), based on Apache 2.0
with additional terms. D-FINE attribution is retained in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and
[its upstream license](third_party/D-FINE/LICENSE). Model weight terms are
provided by the model repository.
