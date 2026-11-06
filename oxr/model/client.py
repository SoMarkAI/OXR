import asyncio
import base64
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from oxr.config.settings import OXRModelConfig, settings
from oxr.model.base import (
    DEFAULT_MODEL_OPTIONS,
    ModelOutput,
    resolve_model_options,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_MODEL_OPTIONS",
    "ModelClient",
    "ModelOutput",
    "OXRModelClient",
    "resolve_model_options",
]


class OXRModelClient:
    """Persistent OpenAI-compatible client used only for OXR requests."""

    def __init__(self, config: Optional[OXRModelConfig] = None):
        """Store the optional config and lazily built HTTP client."""
        self._config = config
        self._client: Optional[AsyncOpenAI] = None
        self._request_loop: Optional[asyncio.AbstractEventLoop] = None
        self._request_limit: Optional[int] = None
        self._request_semaphore: Optional[asyncio.Semaphore] = None

    async def warmup(self) -> None:
        """Remote servers are warmed by their own process; nothing to do here."""

    @property
    def config(self) -> OXRModelConfig:
        """Resolved model configuration for this client."""
        return self._config or settings.oxr_model

    @property
    def client(self) -> Optional[AsyncOpenAI]:
        """Create the OpenAI-compatible client on first use."""
        if self._client is None and self.config.url:
            self._client = AsyncOpenAI(
                base_url=self.config.url,
                api_key="EMPTY",
                max_retries=0,
                http_client=DefaultAsyncHttpxClient(trust_env=False),
            )
        return self._client

    def get_request_semaphore(self) -> asyncio.Semaphore:
        """Return the process client's shared limiter for the current event loop."""
        loop = asyncio.get_running_loop()
        limit = self.config.max_concurrency
        if (
            self._request_loop is not loop
            or self._request_limit != limit
            or self._request_semaphore is None
        ):
            self._request_loop = loop
            self._request_limit = limit
            self._request_semaphore = asyncio.Semaphore(limit)
        assert self._request_semaphore is not None
        return self._request_semaphore

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        try:
            if self._client is not None:
                await self._client.close()
        finally:
            self._client = None
            self._request_loop = None
            self._request_limit = None
            self._request_semaphore = None

    def _encode_image(self, image: np.ndarray) -> str:
        """Encode a BGR page crop as PNG bytes."""
        success, buffer = cv2.imencode(".png", image)
        if not success:
            raise ValueError("Unable to encode image as PNG.")
        return base64.b64encode(buffer).decode("utf-8")

    def _image_to_data_url(self, image: np.ndarray) -> str:
        """Wrap PNG bytes in a data URL."""
        return f"data:image/png;base64,{self._encode_image(image)}"

    async def _call_vision_model(
        self,
        prompt: str,
        image: np.ndarray,
        timeout: int,
        model_options: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send one image and prompt to the model service."""
        if not self.client:
            raise RuntimeError("OXR model URL is not configured.")
        if image is None:
            raise ValueError("An image is required for vision model calls.")

        content = [
            {
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(image)},
            },
            {"type": "text", "text": prompt},
        ]
        resolved_options = resolve_model_options(model_options)

        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": content}],
                timeout=timeout,
                temperature=resolved_options["temperature"],
                top_p=resolved_options["top_p"],
                max_tokens=max_tokens if max_tokens is not None else self.config.max_tokens,
                extra_body={
                    "repetition_penalty": resolved_options["repetition_penalty"],
                },
            ),
            timeout=timeout,
        )
        choice = response.choices[0]
        return ModelOutput(
            choice.message.content or "",
            finish_reason=getattr(choice, "finish_reason", None) or "unknown",
            usage=getattr(response, "usage", None),
        )

    async def request_reading_order(
        self,
        prompt: str,
        image: np.ndarray,
        max_tokens: int,
    ) -> str:
        """Submit a deterministic OXR reading-order request with a prepared prompt."""
        return await self._call_vision_model(
            prompt=prompt,
            image=image,
            timeout=self.config.timeout,
            model_options=None,
            max_tokens=max_tokens,
        )

    async def recognize_text(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        """Recognize body text."""
        from oxr.model.prompts import PROMPT_TEXT_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_TEXT_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_formula(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        """Recognize a formula as LaTeX."""
        from oxr.model.prompts import PROMPT_FORMULA_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_FORMULA_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_table(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        """Recognize a table as HTML."""
        from oxr.model.prompts import PROMPT_TABLE_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_TABLE_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_code(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        """Recognize a code block."""
        from oxr.model.prompts import PROMPT_CODE_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_CODE_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def analyze_layout(self, image: np.ndarray, model_options: Optional[Dict[str, Any]] = None) -> str:
        """Run the diagnostic OXR layout prompt outside the document pipeline."""
        from oxr.model.prompts import PROMPT_LAYOUT_ANALYSIS
        return await self._call_vision_model(
            prompt=PROMPT_LAYOUT_ANALYSIS,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_whole_page(
        self,
        image: np.ndarray,
        model_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Recognize a page without layout analysis."""
        from oxr.model.prompts import PROMPT_WHOLE_PAGE_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_WHOLE_PAGE_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )

    async def recognize_chemical_structure(
        self,
        image: np.ndarray,
        model_options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Recognize a chemical structure as SMILES."""
        from oxr.model.prompts import PROMPT_CHEMICAL_STRUCTURE_RECOGNITION
        return await self._call_vision_model(
            prompt=PROMPT_CHEMICAL_STRUCTURE_RECOGNITION,
            image=image,
            timeout=self.config.timeout,
            model_options=model_options,
        )


ModelClient = OXRModelClient
