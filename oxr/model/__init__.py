"""Recognition backends selected by ``oxr_model.backend``.

``remote``  — an OpenAI-compatible server (vLLM, or a hosted endpoint).
``mlx``     — OXR-1.0 running in this process on Apple Silicon.

Both satisfy :class:`oxr.model.base.OXRModelBackend`; the pipeline only depends on
that interface, so adding a backend means adding a class and a config value.
"""

from __future__ import annotations

from typing import Optional

from oxr.config.settings import OXRModelConfig, settings
from oxr.model.base import ModelOutput, OXRModelBackend
from oxr.model.client import OXRModelClient


def build_model(config: Optional[OXRModelConfig] = None) -> OXRModelBackend:
    """Instantiate the backend named by `oxr_model.backend`."""
    resolved = config or settings.oxr_model
    if resolved.backend == "mlx":
        # Imported lazily so the mlx extra stays optional on CUDA/CPU hosts.
        from oxr.model.local_mlx import LocalMLXModel

        return LocalMLXModel(resolved)
    return OXRModelClient(resolved)


model_client: OXRModelBackend = build_model()

__all__ = ["ModelOutput", "OXRModelBackend", "build_model", "model_client"]
