"""Backend selection and server validation; no optional dependencies required."""

import pytest

from oxr.config.settings import LayoutAnalyzeModelConfig, OXRModelConfig, Settings
from oxr.model import OXRModelBackend, build_model
from oxr.model.client import OXRModelClient


def test_default_backend_is_remote():
    """The default backend is the remote client."""
    backend = build_model(OXRModelConfig(url="http://127.0.0.1:8080/v1"))
    assert isinstance(backend, OXRModelClient)
    assert isinstance(backend, OXRModelBackend)


def test_unknown_backend_is_rejected():
    """An unknown backend name is rejected."""
    with pytest.raises(ValueError):
        OXRModelConfig(backend="tensorrt")


def test_server_validation_requires_url_for_remote():
    """Remote backends need a URL."""
    settings = Settings(
        layout_analyze_model=LayoutAnalyzeModelConfig(url="http://127.0.0.1:8081"),
        oxr_model=OXRModelConfig(backend="remote"),
    )
    with pytest.raises(ValueError, match="oxr_model.url"):
        settings.validate_for_server()


def test_server_validation_requires_path_for_mlx():
    """MLX backends need a checkpoint path."""
    settings = Settings(
        layout_analyze_model=LayoutAnalyzeModelConfig(url="http://127.0.0.1:8081"),
        oxr_model=OXRModelConfig(backend="mlx"),
    )
    with pytest.raises(ValueError, match="oxr_model.model_path"):
        settings.validate_for_server()


def test_server_validation_rejects_missing_mlx_checkpoint(tmp_path):
    """A missing MLX checkpoint is rejected."""
    settings = Settings(
        layout_analyze_model=LayoutAnalyzeModelConfig(url="http://127.0.0.1:8081"),
        oxr_model=OXRModelConfig(
            backend="mlx", model_path=str(tmp_path / "does-not-exist")
        ),
    )
    with pytest.raises(ValueError, match="not a directory"):
        settings.validate_for_server()


def test_server_validation_accepts_existing_mlx_checkpoint(tmp_path):
    """An existing MLX checkpoint passes."""
    settings = Settings(
        layout_analyze_model=LayoutAnalyzeModelConfig(url="http://127.0.0.1:8081"),
        oxr_model=OXRModelConfig(backend="mlx", model_path=str(tmp_path)),
    )
    settings.validate_for_server()
