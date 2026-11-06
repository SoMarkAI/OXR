"""In-process OXR-1.0 recognition on Apple Silicon through MLX.

Implements the same interface as the remote client. Inference runs on a single
worker thread because MLX arrays belong to the thread (and stream) that created
them — in particular the weights, which are loaded on first use *inside* that
thread.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional

import cv2
import numpy as np
import PIL.Image

from oxr.config.settings import OXRModelConfig, settings
from oxr.model.base import ModelOutput, resolve_model_options
from oxr.model.prompts import (
    PROMPT_CHEMICAL_STRUCTURE_RECOGNITION,
    PROMPT_CODE_RECOGNITION,
    PROMPT_FORMULA_RECOGNITION,
    PROMPT_LAYOUT_ANALYSIS,
    PROMPT_TABLE_RECOGNITION,
    PROMPT_TEXT_RECOGNITION,
    PROMPT_WHOLE_PAGE_RECOGNITION,
)

logger = logging.getLogger(__name__)


class LocalMLXModel:
    """OXR-1.0 recognition executed in this process with MLX."""

    def __init__(self, config: Optional[OXRModelConfig] = None):
        """Store the config; the model is loaded on first use."""
        self._config = config
        self._executor: Optional[ThreadPoolExecutor] = None
        self._runner = None
        # One device stream: recognition and reading order share this budget.
        self._semaphore = asyncio.Semaphore(1)

    @property
    def config(self) -> OXRModelConfig:
        """Resolved model configuration for this backend."""
        return self._config or settings.oxr_model

    def get_request_semaphore(self) -> asyncio.Semaphore:
        """One device stream, so one request at a time."""
        return self._semaphore

    def _worker(self) -> ThreadPoolExecutor:
        """Create the single thread that owns the MLX stream."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="oxr-mlx"
            )
        return self._executor

    def _load(self):
        """Runs on the worker thread so the weights share its stream."""
        if self._runner is None:
            from oxr.model.mlx.runner import OXRRunner

            path = self.config.model_path
            if not path:
                raise RuntimeError(
                    "oxr_model.model_path is required when oxr_model.backend is 'mlx'"
                )
            logger.info("loading OXR-1.0 from %s (dtype=%s)", path, self.config.dtype)
            self._runner = OXRRunner(path, self.config.dtype)
        return self._runner

    async def warmup(self) -> None:
        """Load the model during server startup."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(self._worker(), self._load)

    async def _generate(
        self,
        image: np.ndarray,
        prompt: str,
        model_options: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelOutput:
        """Run one prompt off the event loop."""
        options = resolve_model_options(model_options)
        limit = max_tokens or self.config.max_tokens
        loop = asyncio.get_running_loop()

        def work():
            """Blocking call executed on the model's own thread."""
            runner = self._load()
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return runner.generate(
                PIL.Image.fromarray(rgb),
                prompt,
                limit,
                options["repetition_penalty"],
            )

        generation = await loop.run_in_executor(self._worker(), work)
        return ModelOutput(
            generation.text, finish_reason=generation.finish_reason
        )

    async def recognize_text(self, image, model_options=None) -> ModelOutput:
        """Recognize body text."""
        return await self._generate(image, PROMPT_TEXT_RECOGNITION, model_options)

    async def recognize_formula(self, image, model_options=None) -> ModelOutput:
        """Recognize a formula as LaTeX."""
        return await self._generate(image, PROMPT_FORMULA_RECOGNITION, model_options)

    async def recognize_table(self, image, model_options=None) -> ModelOutput:
        """Recognize a table as HTML/Markdown."""
        return await self._generate(image, PROMPT_TABLE_RECOGNITION, model_options)

    async def recognize_code(self, image, model_options=None) -> ModelOutput:
        """Recognize a code block."""
        return await self._generate(image, PROMPT_CODE_RECOGNITION, model_options)

    async def recognize_chemical_structure(self, image, model_options=None) -> ModelOutput:
        """Recognize a structure as SMILES."""
        return await self._generate(
            image, PROMPT_CHEMICAL_STRUCTURE_RECOGNITION, model_options
        )

    async def recognize_whole_page(self, image, model_options=None) -> ModelOutput:
        """Recognize a page without layout analysis."""
        return await self._generate(
            image, PROMPT_WHOLE_PAGE_RECOGNITION, model_options
        )

    async def analyze_layout(self, image, model_options=None) -> ModelOutput:
        """Run the diagnostic layout prompt."""
        return await self._generate(image, PROMPT_LAYOUT_ANALYSIS, model_options)

    async def request_reading_order(
        self, prompt: str, image: np.ndarray, max_tokens: int
    ) -> ModelOutput:
        """Order detected boxes for one page."""
        return await self._generate(image, prompt, max_tokens=max_tokens)

    async def aclose(self) -> None:
        """Stop the worker thread and drop the weights."""
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        self._runner = None
