"""Backend-neutral interface for the OXR recognition model.

The pipeline never talks to a service directly: it calls the methods below on the
backend selected by ``oxr_model.backend``. ``remote`` (an OpenAI-compatible
server) and ``mlx`` (in-process inference on Apple Silicon) implement the same
contract, so nothing above this module needs to know where the model runs.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import numpy as np

DEFAULT_MODEL_OPTIONS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
}


class ModelOutput(str):
    """String-compatible model output with completion metadata attached."""

    finish_reason: str
    usage: Any

    def __new__(
        cls,
        content: str,
        *,
        finish_reason: str = "unknown",
        usage: Any = None,
    ) -> "ModelOutput":
        """Build a string whose completion metadata rides along."""
        instance = super().__new__(cls, content)
        instance.finish_reason = finish_reason
        instance.usage = usage
        return instance


def resolve_model_options(
    model_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Merge caller options over the sampling defaults."""
    resolved = dict(DEFAULT_MODEL_OPTIONS)
    if model_options:
        resolved.update(
            {
                key: float(value)
                for key, value in model_options.items()
                if key in resolved and value is not None
            }
        )
    return resolved


@runtime_checkable
class OXRModelBackend(Protocol):
    """Everything the pipeline needs from a recognition model."""

    async def warmup(self) -> None:
        """Prepare the model before the first request (no-op for remote servers)."""

    def get_request_semaphore(self) -> asyncio.Semaphore:
        """Return the shared in-flight budget for model calls."""

    async def recognize_text(
        self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Recognize body text."""

    async def recognize_formula(
        self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Recognize a formula as LaTeX."""

    async def recognize_table(
        self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Recognize a table as HTML or Markdown."""

    async def recognize_code(
        self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Recognize a code block."""

    async def recognize_chemical_structure(
        self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Recognize a chemical structure as SMILES."""

    async def recognize_whole_page(
        self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Recognize a page without layout analysis."""

    async def analyze_layout(
        self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None
    ) -> ModelOutput:
        """Run the diagnostic layout prompt."""

    async def request_reading_order(
        self, prompt: str, image: np.ndarray, max_tokens: int
    ) -> ModelOutput:
        """Order the detected boxes of one page."""

    async def aclose(self) -> None:
        """Release any resources held by the backend."""
