from pathlib import Path
import copy
import hashlib
import json

import pytest

import dfine_la.config as config
from dfine_la.config import Settings


@pytest.fixture
def model_bundle(tmp_path):
    checkpoint = tmp_path / "last.pth"
    checkpoint.write_bytes(b"checkpoint")
    metadata = copy.deepcopy(config.SUPPORTED_METADATA)
    metadata.update(
        model_version="1.0.0", checkpoint="last.pth",
        checkpoint_size_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata))
    return checkpoint, path, metadata


def test_checkpoint_validation_accepts_expected_file(model_bundle):
    checkpoint, path, _ = model_bundle
    settings = Settings(model_path=checkpoint, device="cpu").validate_checkpoint()
    assert settings.model_version == "1.0.0"
    assert settings.metadata_path == path
    assert settings.id2label["9"] == "Code Block"
    assert settings.device == "cpu"
    assert "metadata_path" not in settings.public_dict()
    assert "model_path" not in settings.public_dict()


def test_checkpoint_validation_rejects_missing_file(tmp_path: Path):
    with pytest.raises(RuntimeError, match="checkpoint not found"):
        Settings(model_path=tmp_path / "missing.pth").validate_checkpoint()


def test_device_from_env(monkeypatch):
    monkeypatch.setenv("DFINE_DEVICE", "cpu")
    assert Settings.from_env().device == "cpu"


def test_invalid_device_from_env(monkeypatch):
    monkeypatch.setenv("DFINE_DEVICE", "metal")
    with pytest.raises(RuntimeError, match="DFINE_DEVICE must be one of"):
        Settings.from_env()


def test_release_checkpoint_defaults(monkeypatch):
    monkeypatch.delenv("MODEL_PATH", raising=False)
    settings = Settings.from_env()
    assert settings.model_path == Path("/models/layout_analysis/last.pth")
    assert settings.model_version == "unknown"  # Resolved from metadata at startup.


def test_model_path_override(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/custom/last.pth")
    assert Settings.from_env().model_path == Path("/custom/last.pth")


def test_checkpoint_validation_rejects_wrong_hash(model_bundle):
    checkpoint, _, _ = model_bundle
    checkpoint.write_bytes(b"checkpoinx")  # Same size; fails the integrity check.
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        Settings(model_path=checkpoint).validate_checkpoint()


def test_checkpoint_validation_rejects_wrong_size(model_bundle):
    checkpoint, _, _ = model_bundle
    checkpoint.write_bytes(b"short")
    with pytest.raises(RuntimeError, match="size mismatch"):
        Settings(model_path=checkpoint).validate_checkpoint()


@pytest.mark.parametrize("field,value", [
    ("schema_version", 2), ("schema_version", True),
    ("model_type", "other"), ("architecture", "stage2_b4"),
    ("image_size", 800), ("num_classes", 13), ("color_order", "bgr"),
    ("resize_backend", "bilinear"), ("postprocess", "other"),
    ("provides_reading_order", True), ("id2label", {"0": "Text"}),
    ("model_version", ""), ("model_version", "training_release"),
    ("model_version", "1.0"), ("checkpoint", "../last.pth"),
    ("checkpoint_size_bytes", True), ("checkpoint_size_bytes", -1),
    ("checkpoint_sha256", "invalid"),
])
def test_rejects_invalid_or_unsupported_metadata(model_bundle, field, value):
    checkpoint, path, metadata = model_bundle
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="metadata"):
        Settings(model_path=checkpoint).validate_checkpoint()


@pytest.mark.parametrize("content", ["not JSON", "[]"])
def test_rejects_malformed_metadata(model_bundle, content):
    checkpoint, path, _ = model_bundle
    path.write_text(content)
    with pytest.raises(RuntimeError, match="metadata"):
        Settings(model_path=checkpoint).validate_checkpoint()


def test_metadata_is_required(model_bundle):
    checkpoint, path, _ = model_bundle
    path.unlink()
    with pytest.raises(RuntimeError, match="metadata"):
        Settings(model_path=checkpoint).validate_checkpoint()


def test_metadata_path_override(model_bundle, monkeypatch):
    checkpoint, path, _ = model_bundle
    override = path.with_name("custom.json")
    path.rename(override)
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("MODEL_METADATA_PATH", str(override))
    assert Settings.from_env().validate_checkpoint().metadata_path == override


def test_missing_required_metadata_field(model_bundle):
    checkpoint, path, metadata = model_bundle
    metadata.pop("image_size")
    path.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="image_size"):
        Settings(model_path=checkpoint).validate_checkpoint()


def test_metadata_does_not_override_runtime_settings(model_bundle):
    checkpoint, path, metadata = model_bundle
    metadata.update(device="cuda", inference_precision="fp16_autocast",
                    cuda_graph_enabled=True)
    path.write_text(json.dumps(metadata))
    settings = Settings(model_path=checkpoint, device="cpu",
                        inference_precision="fp32", cuda_graph_enabled=False)
    loaded = settings.validate_checkpoint()
    assert loaded.device == "cpu"
    assert loaded.inference_precision == "fp32"
    assert loaded.cuda_graph_enabled is False
