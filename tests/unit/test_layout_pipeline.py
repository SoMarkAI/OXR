import asyncio

import numpy as np
import pytest

from oxr.config.settings import LayoutAnalyzeModelConfig, settings
from oxr.pipeline.errors import PipelineError
from oxr.pipeline.layout import analyze_layouts


def response_for(encoded_images):
    labels = []
    boxes = []
    scores = []
    for image in encoded_images:
        marker = image[0] if len(image) == 1 else 0
        labels.append([3])
        boxes.append([[marker, 1, marker + 1, 2]])
        scores.append([0.99])
    return {
        "status": "success",
        "params": {},
        "data": {"labels": labels, "boxes": boxes, "scores": scores},
    }


def test_analyze_layouts_passes_configured_score_threshold(monkeypatch):
    captured = {}

    def encode_images(_images):
        return [b"page"]

    async def request_batch(encoded_images):
        return response_for(encoded_images)

    def parse_layout_batch(payload, image_sizes, score_threshold, nms_iou_threshold):
        captured["payload"] = payload
        captured["image_sizes"] = image_sizes
        captured["score_threshold"] = score_threshold
        captured["nms_iou_threshold"] = nms_iou_threshold
        return [[]]

    monkeypatch.setattr("oxr.pipeline.layout.layout_client.encode_images", encode_images)
    monkeypatch.setattr("oxr.pipeline.layout.layout_client.request_batch", request_batch)
    monkeypatch.setattr("oxr.pipeline.layout.parse_layout_batch", parse_layout_batch)
    monkeypatch.setattr(settings.layout_analyze_model, "score_threshold", 0.73)
    monkeypatch.setattr(settings.layout_analyze_model, "retry_times", 1)

    result = asyncio.run(analyze_layouts([np.zeros((8, 9, 3), dtype=np.uint8)]))

    assert result == [[]]
    assert captured["image_sizes"] == [(9, 8)]
    assert captured["score_threshold"] == 0.73
    assert captured["nms_iou_threshold"] == 0.6


def test_analyze_layouts_batches_concurrently_and_restores_document_order(monkeypatch):
    images = [np.full((10, 10, 3), index, dtype=np.uint8) for index in range(5)]
    calls = []

    def encode_images(batch):
        return [bytes([int(image[0, 0, 0])]) for image in batch]

    async def request_batch(encoded_images):
        calls.append(tuple(encoded_images))
        await asyncio.sleep(0.01 if encoded_images[0] == b"\x00" else 0)
        return response_for(encoded_images)

    monkeypatch.setattr("oxr.pipeline.layout.layout_client.encode_images", encode_images)
    monkeypatch.setattr("oxr.pipeline.layout.layout_client.request_batch", request_batch)
    monkeypatch.setattr(settings.layout_analyze_model, "batch_size", 2)
    monkeypatch.setattr(settings.layout_analyze_model, "max_concurrency", 3)
    monkeypatch.setattr(settings.layout_analyze_model, "retry_times", 1)

    blocks = asyncio.run(analyze_layouts(images))

    assert sorted(calls) == [(b"\x00", b"\x01"), (b"\x02", b"\x03"), (b"\x04",)]
    assert len(blocks) == 5
    assert [page[0].bbox for page in blocks] == [
        [0.0, 1.0, 1.0, 2.0],
        [1.0, 1.0, 2.0, 2.0],
        [2.0, 1.0, 3.0, 2.0],
        [3.0, 1.0, 4.0, 2.0],
        [4.0, 1.0, 5.0, 2.0],
    ]


def test_analyze_layouts_encodes_once_and_retries_whole_batch(monkeypatch):
    images = [np.zeros((10, 10, 3), dtype=np.uint8), np.ones((10, 10, 3), dtype=np.uint8)]
    encode_calls = 0
    requests = []

    def encode_images(batch):
        nonlocal encode_calls
        encode_calls += 1
        return [b"first", b"second"]

    async def request_batch(encoded_images):
        requests.append(tuple(encoded_images))
        if len(requests) == 1:
            raise RuntimeError("transient layout failure")
        return response_for(encoded_images)

    monkeypatch.setattr("oxr.pipeline.layout.layout_client.encode_images", encode_images)
    monkeypatch.setattr("oxr.pipeline.layout.layout_client.request_batch", request_batch)
    monkeypatch.setattr(settings.layout_analyze_model, "batch_size", 2)
    monkeypatch.setattr(settings.layout_analyze_model, "retry_times", 2)
    original_sleep = asyncio.sleep
    monkeypatch.setattr("oxr.pipeline.layout.asyncio.sleep", lambda _seconds: original_sleep(0))

    assert len(asyncio.run(analyze_layouts(images))) == 2
    assert encode_calls == 1
    assert requests == [(b"first", b"second"), (b"first", b"second")]


def test_request_semaphore_is_limited_and_released_during_backoff(monkeypatch):
    images = [np.full((10, 10, 3), index, dtype=np.uint8) for index in range(2)]
    active = 0
    peak = 0
    first_batch_attempts = 0
    second_request_started = asyncio.Event()

    def encode_images(batch):
        return [bytes([int(image[0, 0, 0])]) for image in batch]

    async def request_batch(encoded_images):
        nonlocal active, peak, first_batch_attempts
        active += 1
        peak = max(peak, active)
        try:
            if encoded_images == [b"\x00"]:
                first_batch_attempts += 1
                if first_batch_attempts == 1:
                    raise RuntimeError("retry me")
            second_request_started.set()
            return response_for(encoded_images)
        finally:
            active -= 1

    original_sleep = asyncio.sleep

    async def immediate_backoff(_seconds):
        await original_sleep(0)
        assert second_request_started.is_set()

    monkeypatch.setattr("oxr.pipeline.layout.layout_client.encode_images", encode_images)
    monkeypatch.setattr("oxr.pipeline.layout.layout_client.request_batch", request_batch)
    monkeypatch.setattr(settings.layout_analyze_model, "batch_size", 1)
    monkeypatch.setattr(settings.layout_analyze_model, "max_concurrency", 1)
    monkeypatch.setattr(settings.layout_analyze_model, "retry_times", 2)
    monkeypatch.setattr("oxr.pipeline.layout.asyncio.sleep", immediate_backoff)

    assert len(asyncio.run(analyze_layouts(images))) == 2
    assert peak == 1


def test_analyze_layouts_cancels_and_drains_sibling_batches_on_failure(monkeypatch):
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    def encode_images(images):
        return [bytes([int(image[0, 0, 0])]) for image in images]

    async def request_batch(encoded_images):
        if encoded_images == [b"\x00"]:
            await sibling_started.wait()
            raise RuntimeError("failed batch")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    monkeypatch.setattr("oxr.pipeline.layout.layout_client.encode_images", encode_images)
    monkeypatch.setattr("oxr.pipeline.layout.layout_client.request_batch", request_batch)
    monkeypatch.setattr(settings.layout_analyze_model, "batch_size", 1)
    monkeypatch.setattr(settings.layout_analyze_model, "max_concurrency", 2)
    monkeypatch.setattr(settings.layout_analyze_model, "retry_times", 1)

    async def scenario():
        images = [
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.ones((8, 8, 3), dtype=np.uint8),
        ]
        with pytest.raises(PipelineError, match="failed batch"):
            await analyze_layouts(images)
        assert sibling_cancelled.is_set()

    asyncio.run(scenario())


def test_analyze_layouts_uses_injected_client_and_config():
    calls = []
    config = LayoutAnalyzeModelConfig(
        url="http://layout.example",
        batch_size=1,
        max_concurrency=1,
        retry_times=1,
        score_threshold=0.8,
    )

    class Client:
        def encode_images(self, images):
            assert len(images) == 1
            return [b"injected"]

        async def request_batch(self, encoded_images):
            calls.append(encoded_images)
            return {
                "status": "success",
                "params": {},
                "data": {
                    "labels": [[3, 7]],
                    "boxes": [[[0, 0, 5, 5], [5, 5, 10, 10]]],
                    "scores": [[0.79, 0.8]],
                },
            }

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    pages = asyncio.run(analyze_layouts([image], client=Client(), config=config))

    assert calls == [[b"injected"]]
    assert len(pages[0]) == 1
    assert pages[0][0].type.value == "Title"
