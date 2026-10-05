from pathlib import Path

import torch

from dfine_la.config import Settings
import dfine_la.runtime as runtime_module


class FakeModel:
    def __call__(self, images, sizes):
        batch_size = images.shape[0]
        return (
            torch.zeros((batch_size, 1), dtype=torch.int64, device=images.device),
            torch.zeros((batch_size, 1, 4), device=images.device),
            torch.ones((batch_size, 1), device=images.device),
        )


def test_cpu_runtime_uses_fp32_eager_inference(monkeypatch):
    monkeypatch.setattr(Settings, "validate_checkpoint", lambda self: self)
    monkeypatch.setattr(runtime_module, "load_model", lambda *args, **kwargs: FakeModel())
    settings = Settings(
        model_path=Path("unused.pth"),
        device="cpu",
        cuda_graph_enabled=True,
    )

    runtime = runtime_module.ModelRuntime(settings)
    outputs, _ = runtime.infer(
        (torch.zeros((1, 3, 8, 8)), torch.tensor([[8, 8]]))
    )

    assert runtime.device == torch.device("cpu")
    assert runtime.cuda_graph_enabled is False
    assert runtime.health()["inference_precision"] == "fp32"
    assert "model_path" not in runtime.health()
    assert "model_path" not in runtime.info()["model_config"]
    assert "metadata_path" not in runtime.info()["model_config"]
    assert runtime.info()["devices"] == ["cpu"]
    assert runtime.info()["model_config"]["device"] == "cpu"
    assert runtime.info()["model_config"]["cuda_graph_enabled"] is False
    assert outputs[0].shape == (1, 1)


def test_cuda_mode_still_requires_cuda(monkeypatch):
    monkeypatch.setattr(Settings, "validate_checkpoint", lambda self: self)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    settings = Settings(model_path=Path("unused.pth"), device="cuda")

    try:
        runtime_module.ModelRuntime(settings)
    except RuntimeError as exc:
        assert "NVIDIA GPU" in str(exc)
    else:
        raise AssertionError("CUDA mode must fail when CUDA is unavailable")
