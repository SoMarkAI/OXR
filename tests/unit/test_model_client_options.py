import asyncio
import base64
from types import SimpleNamespace

import cv2
import numpy as np

from oxr.config.settings import OXRModelConfig
from oxr.model.client import (
    DEFAULT_MODEL_OPTIONS,
    ModelOutput,
    OXRModelClient,
    resolve_model_options,
)
from oxr.model.prompts import (
    PROMPT_CHEMICAL_STRUCTURE_RECOGNITION,
    PROMPT_LAYOUT_ANALYSIS,
    PROMPT_WHOLE_PAGE_RECOGNITION,
)


class FakeCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content="ok")
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage={"completion_tokens": 1})


class FakeClient:
    def __init__(self):
        self.completions = FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)


def test_resolve_model_options_uses_client_defaults():
    assert resolve_model_options() == DEFAULT_MODEL_OPTIONS


def test_oxr_client_disables_openai_sdk_retries(monkeypatch):
    captured = {}
    fake_client = object()
    fake_http_client = object()

    def build_client(**kwargs):
        captured.update(kwargs)
        return fake_client

    monkeypatch.setattr("oxr.model.client.AsyncOpenAI", build_client)
    monkeypatch.setattr(
        "oxr.model.client.DefaultAsyncHttpxClient",
        lambda **_kwargs: fake_http_client,
    )

    client = OXRModelClient(OXRModelConfig(url="http://oxr.example/v1"))

    assert client.client is fake_client
    assert captured["max_retries"] == 0


def test_call_vision_model_passes_sampling_options():
    fake_client = FakeClient()
    client = OXRModelClient()
    client._client = fake_client
    options = {"temperature": 0.4, "top_p": 0.8, "repetition_penalty": 1.3}

    result = asyncio.run(
        client._call_vision_model(
            prompt="prompt",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            timeout=1,
            model_options=options,
        )
    )

    kwargs = fake_client.completions.kwargs
    assert result == "ok"
    assert isinstance(result, ModelOutput)
    assert result.finish_reason == "stop"
    assert result.usage == {"completion_tokens": 1}
    assert kwargs["temperature"] == 0.4
    assert kwargs["top_p"] == 0.8
    assert kwargs["extra_body"]["repetition_penalty"] == 1.3
    assert "chat_template_kwargs" not in kwargs["extra_body"]
    content = kwargs["messages"][0]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert content[1] == {"type": "text", "text": "prompt"}


def test_vision_requests_use_lossless_png_images():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    client = OXRModelClient()

    data_url = client._image_to_data_url(image)

    prefix, encoded = data_url.split(",", 1)
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert prefix == "data:image/png;base64"
    assert np.array_equal(decoded, image)


def test_reading_order_uses_deterministic_options_and_max_token_override():
    fake_client = FakeClient()
    client = OXRModelClient()
    client._client = fake_client

    result = asyncio.run(
        client.request_reading_order(
            prompt="reading-order prompt",
            image=np.zeros((8, 8, 3), dtype=np.uint8),
            max_tokens=24,
        )
    )

    kwargs = fake_client.completions.kwargs
    assert result == "ok"
    assert kwargs["temperature"] == 0.0
    assert kwargs["top_p"] == 1.0
    assert kwargs["extra_body"]["repetition_penalty"] == 1.0
    assert kwargs["max_tokens"] == 24


def test_oxr_client_closes_persistent_client():
    class ClosableFakeClient(FakeClient):
        def __init__(self):
            super().__init__()
            self.closed = False

        async def close(self):
            self.closed = True

    fake_client = ClosableFakeClient()
    client = OXRModelClient()
    client._client = fake_client

    asyncio.run(client.aclose())

    assert fake_client.closed is True
    assert client._client is None


def test_diagnostic_client_methods_use_central_prompt_definitions():
    calls = []
    client = OXRModelClient(OXRModelConfig(url="http://oxr.example/v1", timeout=17))
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    options = {"temperature": 0.2}

    async def call(prompt, image, timeout, model_options=None, max_tokens=None):
        calls.append((prompt, image, timeout, model_options, max_tokens))
        return "ok"

    client._call_vision_model = call

    async def scenario():
        assert await client.analyze_layout(image, options) == "ok"
        assert await client.recognize_whole_page(image, options) == "ok"
        assert await client.recognize_chemical_structure(image, options) == "ok"

    asyncio.run(scenario())

    assert [entry[0] for entry in calls] == [
        PROMPT_LAYOUT_ANALYSIS,
        PROMPT_WHOLE_PAGE_RECOGNITION,
        PROMPT_CHEMICAL_STRUCTURE_RECOGNITION,
    ]
    assert all(entry[1] is image for entry in calls)
    assert all(entry[2] == 17 for entry in calls)
    assert all(entry[3] == options for entry in calls)
