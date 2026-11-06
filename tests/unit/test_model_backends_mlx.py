"""The in-process MLX backend; skipped unless the optional extra is installed."""

import asyncio

import numpy as np
import pytest

pytest.importorskip("mlx.core", reason="requires the 'mlx' extra")
pytest.importorskip("transformers", reason="requires the 'mlx' extra")
pytest.importorskip("PIL.Image", reason="requires the 'mlx' extra")

from oxr.config.settings import OXRModelConfig
from oxr.model import OXRModelBackend, build_model
from oxr.model.base import ModelOutput
from oxr.model.local_mlx import LocalMLXModel
from oxr.model.mlx.runner import Generation, finish_reason_for
from oxr.model.prompts import (
    PROMPT_CODE_RECOGNITION,
    PROMPT_FORMULA_RECOGNITION,
    PROMPT_TABLE_RECOGNITION,
    PROMPT_TEXT_RECOGNITION,
)

IMAGE = np.zeros((8, 8, 3), dtype=np.uint8)


class FakeRunner:
    def __init__(self, finish_reason: str = "length"):
        """Record calls and return a fixed finish reason."""
        self.calls = []
        self.finish_reason = finish_reason

    def generate(self, image, prompt, max_tokens, repetition_penalty):
        """Act as the MLX runner without loading weights."""
        self.calls.append((prompt, max_tokens, repetition_penalty))
        return Generation(text="ok", finish_reason=self.finish_reason)


def _local(tmp_path, finish_reason: str = "length") -> tuple[LocalMLXModel, FakeRunner]:
    """Build a local backend whose loader returns a fake runner."""
    model = LocalMLXModel(OXRModelConfig(backend="mlx", model_path=str(tmp_path)))
    runner = FakeRunner(finish_reason)
    model._load = lambda: runner
    return model, runner


def test_mlx_backend_selected_by_config(tmp_path):
    """The factory returns the MLX backend."""
    backend = build_model(OXRModelConfig(backend="mlx", model_path=str(tmp_path)))
    assert isinstance(backend, LocalMLXModel)
    assert isinstance(backend, OXRModelBackend)


@pytest.mark.parametrize(
    ("method", "prompt"),
    [
        ("recognize_text", PROMPT_TEXT_RECOGNITION),
        ("recognize_formula", PROMPT_FORMULA_RECOGNITION),
        ("recognize_table", PROMPT_TABLE_RECOGNITION),
        ("recognize_code", PROMPT_CODE_RECOGNITION),
    ],
)
def test_local_backend_routes_prompts(tmp_path, method, prompt):
    """Each recognition method sends its own prompt."""
    model, runner = _local(tmp_path)
    result = asyncio.run(getattr(model, method)(IMAGE))
    assert isinstance(result, ModelOutput)
    assert result == "ok"
    assert runner.calls[0][0] == prompt


def test_local_backend_reports_finish_reason(tmp_path):
    """Finish reason is propagated to ModelOutput."""
    model, _ = _local(tmp_path, finish_reason="stop")
    result = asyncio.run(model.recognize_text(IMAGE))
    assert result.finish_reason == "stop"


def test_local_backend_honours_max_tokens(tmp_path):
    """Requested token limits reach the runner."""
    model, runner = _local(tmp_path)
    asyncio.run(model.request_reading_order("order the boxes", IMAGE, 64))
    assert runner.calls[0][1] == 64


def test_local_backend_serialises_requests(tmp_path):
    """Only one request runs at a time."""
    model, _ = _local(tmp_path)
    assert model.get_request_semaphore()._value == 1


@pytest.mark.parametrize("eos_id", [151643, 151645])
def test_runner_reports_stop_on_eos(eos_id):
    """EOS tokens end generation cleanly."""
    assert finish_reason_for(eos_id) == "stop"


def test_runner_reports_length_when_cap_ends_generation():
    """A hard cap reports truncation."""
    assert finish_reason_for(1234) == "length"
