import asyncio

import cv2
import httpx
import numpy as np
import pytest

from oxr.config.settings import LayoutAnalyzeModelConfig
from oxr.model.layout_client import LayoutAnalyzeClient


def test_encode_images_returns_lossless_pngs():
    client = LayoutAnalyzeClient(LayoutAnalyzeModelConfig(url="http://layout:8081"))
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)

    encoded = client.encode_images([image])

    assert len(encoded) == 1
    assert encoded[0].startswith(b"\x89PNG\r\n\x1a\n")
    decoded = cv2.imdecode(np.frombuffer(encoded[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert np.array_equal(decoded, image)


def test_request_batch_posts_repeated_images_to_inference_endpoint():
    requests = []
    native_response = {
        "status": "success",
        "params": {},
        "data": {"labels": [[], []], "boxes": [[], []], "scores": [[], []]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=native_response)

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport, base_url="http://layout:8081")
    client = LayoutAnalyzeClient(
        LayoutAnalyzeModelConfig(url="http://layout:8081/"),
        http_client=http_client,
    )

    result = asyncio.run(client.request_batch([b"first", b"second"]))
    asyncio.run(client.aclose())

    assert result == native_response
    assert requests[0].url == httpx.URL("http://layout:8081/inference")
    assert requests[0].content.count(b'name="images"') == 2
    assert requests[0].content.count(b".png") == 2
    assert requests[0].content.count(b"application/octet-stream") == 2
    assert http_client.is_closed is True


@pytest.mark.parametrize(
    "configured_url",
    [
        "http://layout:8081/v1/layout/analyze",
        "http://layout:8081/v1/layout/analyze/",
    ],
)
def test_request_batch_replaces_legacy_layout_endpoint_with_inference(configured_url):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://layout:8081",
    )
    client = LayoutAnalyzeClient(
        LayoutAnalyzeModelConfig(url=configured_url),
        http_client=http_client,
    )

    asyncio.run(client.request_batch([b"page"]))
    asyncio.run(client.aclose())

    assert requests[0].url == httpx.URL("http://layout:8081/inference")
