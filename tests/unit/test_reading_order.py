import asyncio

import numpy as np
import pytest

import oxr.pipeline.reading_order as reading_order_module
from oxr.config.settings import OXRModelConfig, settings
from oxr.model.prompts import build_reading_order_prompt
from oxr.pipeline.errors import PipelineError
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.reading_order import (
    parse_reading_order,
    repair_incomplete_reading_order,
    resolve_reading_orders,
    supplement_incomplete_reading_order,
)


def _block(idx: int, block_type: BlockType, bbox: list[float]) -> Block:
    return Block(idx, block_type, bbox, "", ContentFormat.TEXT)


def test_build_reading_order_prompt_matches_oxr_model_pixel_normalization():
    prompt = build_reading_order_prompt(
        {
            "ref": [BlockType.TITLE.value, BlockType.TEXT.value],
            "bbox": [[10, 20, 110, 80], [0, 0, 200, 100]],
            "bbox_type": "real",
        },
        image_size=(200, 100),
    )

    assert prompt == """
Given the detected layout boxes below, output the reading order as a whitespace-separated permutation of 0-based box IDs only.

Boxes:
0 | <|object_ref_start|>Title<|object_ref_end|> | <|box_start|>(50,200),(550,800)<|box_end|>
1 | <|object_ref_start|>Text<|object_ref_end|> | <|box_start|>(0,0),(1000,1000)<|box_end|>

Output only the box IDs."""


@pytest.mark.parametrize(
    ("raw_text", "block_count", "expected"),
    [
        ("", 0, []),
        (" \n\t", 0, []),
        ("0", 1, [0]),
        ("2  0\n1\t3", 4, [2, 0, 1, 3]),
        ("2\r\n0\u20031", 3, [2, 0, 1]),
    ],
)
def test_parse_reading_order_accepts_only_whitespace_separated_permutations(
    raw_text,
    block_count,
    expected,
):
    assert parse_reading_order(raw_text, block_count) == expected


@pytest.mark.parametrize(
    ("raw_text", "block_count"),
    [
        ("0, 1", 2),
        ("order: 0 1", 2),
        ("```\n0 1\n```", 2),
        ("0 0", 2),
        ("0", 2),
        ("0 1 2", 2),
        ("0 2", 2),
        ("-1 0", 2),
    ],
)
def test_parse_reading_order_rejects_invalid_syntax_or_non_permutations(raw_text, block_count):
    with pytest.raises(ValueError, match="reading order"):
        parse_reading_order(raw_text, block_count)


def test_repair_incomplete_reading_order_preserves_model_order():
    blocks = [
        _block(0, BlockType.TEXT, [50, 50, 80, 80]),
        _block(1, BlockType.TEXT, [10, 10, 40, 30]),
        _block(2, BlockType.TEXT, [50, 10, 90, 30]),
    ]

    assert repair_incomplete_reading_order("1 0", blocks) == [1, 2, 0]


def test_repair_incomplete_reading_order_accepts_two_of_many_missing():
    blocks = [
        _block(index, BlockType.TEXT, [0, index * 10, 10, (index + 1) * 10])
        for index in range(20)
    ]
    partial_order = " ".join(str(index) for index in range(20) if index not in {3, 17})

    assert repair_incomplete_reading_order(partial_order, blocks) == list(range(20))


def test_xycut_supplement_preserves_oxr_order_and_collapses_duplicates(monkeypatch):
    blocks = [
        _block(index, BlockType.TEXT, [0, index * 10, 10, (index + 1) * 10])
        for index in range(4)
    ]
    monkeypatch.setattr(
        reading_order_module,
        "xycut_plus_order",
        lambda _blocks: [0, 1, 2, 3],
    )

    supplemented = supplement_incomplete_reading_order("3 1 1", blocks)

    assert supplemented == [0, 2, 3, 1]
    assert [block_id for block_id in supplemented if block_id in {3, 1}] == [3, 1]


def test_xycut_supplement_rejects_output_with_less_than_half_coverage():
    blocks = [
        _block(index, BlockType.TEXT, [0, index * 10, 10, (index + 1) * 10])
        for index in range(5)
    ]

    assert supplement_incomplete_reading_order("1 1", blocks) is None


def test_xycut_supplement_rejects_out_of_range_ids():
    blocks = [
        _block(index, BlockType.TEXT, [0, index * 10, 10, (index + 1) * 10])
        for index in range(4)
    ]

    assert supplement_incomplete_reading_order("0 1 4", blocks) is None


def test_xycut_supplement_handles_dense_page_without_reordering_oxr_ids():
    blocks = [
        _block(index, BlockType.TEXT, [0, index * 4, 100, index * 4 + 3])
        for index in range(300)
    ]
    oxr_order = list(range(180))
    raw_order = " ".join(str(index) for index in [*oxr_order, *range(100)])

    supplemented = supplement_incomplete_reading_order(raw_order, blocks)

    assert supplemented is not None
    assert sorted(supplemented) == list(range(300))
    assert [block_id for block_id in supplemented if block_id < 180] == oxr_order


@pytest.mark.parametrize("raw_text", ["0 0", "0 3", "0", "order: 0 1"])
def test_repair_incomplete_reading_order_rejects_ambiguous_output(raw_text):
    blocks = [
        _block(0, BlockType.TEXT, [0, 0, 10, 10]),
        _block(1, BlockType.TEXT, [0, 10, 10, 20]),
        _block(2, BlockType.TEXT, [0, 20, 10, 30]),
    ]

    assert repair_incomplete_reading_order(raw_text, blocks) is None


def test_resolve_reading_orders_fast_paths_zero_and_one_block_without_oxr(monkeypatch):
    async def unexpected_request(*_args, **_kwargs):
        raise AssertionError("OXR must not be called for zero or one block")

    monkeypatch.setattr(
        "oxr.pipeline.reading_order.model_client.request_reading_order",
        unexpected_request,
    )

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    pages = [[], [_block(0, BlockType.TEXT, [10, 20, 110, 80])]]
    result = asyncio.run(resolve_reading_orders(pages, [image, image], asyncio.Semaphore(1)))

    assert result == [[], [0]]


def test_resolve_reading_orders_uses_original_image_labels_token_cap_and_retries(monkeypatch):
    calls = []
    original_image = np.zeros((100, 200, 3), dtype=np.uint8)
    blocks = [
        _block(9, BlockType.TITLE, [10, 20, 110, 80]),
        _block(4, BlockType.TEXT, [0, 0, 200, 100]),
    ]

    async def request(prompt, image, max_tokens):
        calls.append((prompt, image, max_tokens))
        if len(calls) == 1:
            raise RuntimeError("retry me")
        return "1 0"

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(settings.oxr_model, "retry_times", 2)
    monkeypatch.setattr(settings.oxr_model, "max_tokens", 12)
    monkeypatch.setattr("oxr.pipeline.reading_order.model_client.request_reading_order", request)
    monkeypatch.setattr("oxr.pipeline.reading_order.asyncio.sleep", no_delay)

    result = asyncio.run(resolve_reading_orders([blocks], [original_image], asyncio.Semaphore(1)))

    assert result == [[1, 0]]
    assert len(calls) == 2
    assert all(image is original_image for _prompt, image, _max_tokens in calls)
    assert all(max_tokens == 12 for _prompt, _image, max_tokens in calls)
    assert (
        "0 | <|object_ref_start|>Title<|object_ref_end|> | "
        "<|box_start|>(50,200),(550,800)<|box_end|>"
    ) in calls[0][0]
    assert (
        "1 | <|object_ref_start|>Text<|object_ref_end|> | "
        "<|box_start|>(0,0),(1000,1000)<|box_end|>"
    ) in calls[0][0]


def test_reading_order_token_cap_has_sixteen_token_floor(monkeypatch):
    captured = {}

    async def request(_prompt, _image, max_tokens):
        captured["max_tokens"] = max_tokens
        return "0 1"

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(settings.oxr_model, "max_tokens", 4096)
    monkeypatch.setattr("oxr.pipeline.reading_order.model_client.request_reading_order", request)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    blocks = [
        _block(0, BlockType.TEXT, [0, 0, 5, 5]),
        _block(1, BlockType.TEXT, [5, 5, 10, 10]),
    ]
    asyncio.run(resolve_reading_orders([blocks], [image], asyncio.Semaphore(1)))

    assert captured["max_tokens"] == 16


def test_resolve_reading_orders_uses_injected_client_and_config():
    calls = []
    config = OXRModelConfig(
        url="http://oxr.example/v1",
        max_tokens=9,
        retry_times=1,
    )

    class Client:
        async def request_reading_order(self, prompt, image, max_tokens):
            calls.append((prompt, image, max_tokens))
            return "1 0"

    image = np.zeros((100, 200, 3), dtype=np.uint8)
    blocks = [
        _block(0, BlockType.TITLE, [0, 0, 100, 50]),
        _block(1, BlockType.TEXT, [0, 50, 100, 100]),
    ]
    result = asyncio.run(
        resolve_reading_orders(
            [blocks],
            [image],
            asyncio.Semaphore(1),
            client=Client(),
            config=config,
        )
    )

    assert result == [[1, 0]]
    assert len(calls) == 1
    assert calls[0][1] is image
    assert calls[0][2] == 9


def test_invalid_reading_order_falls_back_to_xycut_without_logging_raw_text(
    caplog,
    monkeypatch,
):
    calls = []

    async def request(_prompt, _image, _max_tokens):
        calls.append(1)
        return "0 0 4"

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(settings.oxr_model, "retry_times", 2)
    monkeypatch.setattr(
        "oxr.pipeline.reading_order.model_client.request_reading_order",
        request,
    )
    monkeypatch.setattr(
        reading_order_module,
        "xycut_plus_order",
        lambda _blocks: [2, 1, 0],
    )
    monkeypatch.setattr("oxr.pipeline.reading_order.asyncio.sleep", no_delay)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    blocks = [
        _block(0, BlockType.TEXT, [0, 0, 3, 10]),
        _block(1, BlockType.TEXT, [3, 0, 6, 10]),
        _block(2, BlockType.TEXT, [6, 0, 10, 10]),
    ]

    with caplog.at_level("WARNING", logger="oxr.pipeline.reading_order"):
        result = asyncio.run(
            resolve_reading_orders([blocks], [image], asyncio.Semaphore(1))
        )

    assert result == [[2, 1, 0]]
    assert len(calls) == 2
    assert "reading_order_xycut_fallback" in caplog.text
    assert "missing_ids=[1, 2]" in caplog.text
    assert "duplicate_ids=[0]" in caplog.text
    assert "out_of_range_ids=[4]" in caplog.text
    assert "raw_sha256=" in caplog.text
    assert "raw_text=" not in caplog.text


@pytest.mark.parametrize("raw_text", ["order: 0 1", "0"])
def test_unusable_reading_order_uses_xycut_only_on_final_attempt(
    raw_text,
    caplog,
    monkeypatch,
):
    calls = []
    config = OXRModelConfig(
        url="http://oxr.example/v1",
        max_tokens=32,
        retry_times=2,
    )

    class Client:
        async def request_reading_order(self, _prompt, _image, _max_tokens):
            calls.append(1)
            return raw_text

    monkeypatch.setattr(
        reading_order_module,
        "xycut_plus_order",
        lambda _blocks: [2, 0, 1],
    )

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(reading_order_module.asyncio, "sleep", no_delay)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    blocks = [
        _block(index, BlockType.TEXT, [0, index * 10, 10, (index + 1) * 10])
        for index in range(3)
    ]

    with caplog.at_level("WARNING", logger="oxr.pipeline.reading_order"):
        result = asyncio.run(
            resolve_reading_orders(
                [blocks],
                [image],
                asyncio.Semaphore(1),
                client=Client(),
                config=config,
            )
        )

    assert result == [[2, 0, 1]]
    assert len(calls) == 2
    assert caplog.text.count("reading_order_invalid") == 1
    assert caplog.text.count("reading_order_xycut_fallback") == 1


def test_reading_order_request_failure_does_not_fall_back_to_xycut(monkeypatch):
    calls = []
    xycut_calls = []

    async def request(_prompt, _image, _max_tokens):
        calls.append(1)
        raise TimeoutError("model unavailable")

    async def no_delay(_seconds):
        return None

    def xycut(_blocks):
        xycut_calls.append(1)
        return [0, 1]

    monkeypatch.setattr(settings.oxr_model, "retry_times", 2)
    monkeypatch.setattr(
        "oxr.pipeline.reading_order.model_client.request_reading_order",
        request,
    )
    monkeypatch.setattr(reading_order_module, "xycut_plus_order", xycut)
    monkeypatch.setattr(reading_order_module.asyncio, "sleep", no_delay)
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    blocks = [
        _block(0, BlockType.TEXT, [0, 0, 5, 10]),
        _block(1, BlockType.TEXT, [5, 0, 10, 10]),
    ]

    with pytest.raises(PipelineError, match="Reading order failed"):
        asyncio.run(resolve_reading_orders([blocks], [image], asyncio.Semaphore(1)))

    assert len(calls) == 2
    assert xycut_calls == []


def test_incomplete_reading_order_is_repaired_without_retry(caplog):
    calls = []
    config = OXRModelConfig(
        url="http://oxr.example/v1",
        max_tokens=32,
        retry_times=2,
    )

    class Client:
        async def request_reading_order(self, _prompt, _image, _max_tokens):
            calls.append(1)
            return "1 0"

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    blocks = [
        _block(0, BlockType.TEXT, [50, 50, 80, 80]),
        _block(1, BlockType.TEXT, [10, 10, 40, 30]),
        _block(2, BlockType.TEXT, [50, 10, 90, 30]),
    ]

    with caplog.at_level("WARNING", logger="oxr.pipeline.reading_order"):
        result = asyncio.run(
            resolve_reading_orders(
                [blocks],
                [image],
                asyncio.Semaphore(1),
                client=Client(),
                config=config,
            )
        )

    assert result == [[1, 2, 0]]
    assert len(calls) == 1
    assert "reading_order_repaired" in caplog.text
    assert "missing_ids=[2]" in caplog.text
    assert "raw_text=" not in caplog.text


def test_broad_xycut_supplement_runs_only_after_final_retry(caplog, monkeypatch):
    calls = []
    config = OXRModelConfig(
        url="http://oxr.example/v1",
        max_tokens=32,
        retry_times=2,
    )

    class Client:
        async def request_reading_order(self, _prompt, _image, _max_tokens):
            calls.append(1)
            return "3 1 1"

    monkeypatch.setattr(
        reading_order_module,
        "xycut_plus_order",
        lambda _blocks: [0, 1, 2, 3],
    )

    async def no_delay(_seconds):
        return None

    monkeypatch.setattr(reading_order_module.asyncio, "sleep", no_delay)
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    blocks = [
        _block(index, BlockType.TEXT, [0, index * 10, 10, (index + 1) * 10])
        for index in range(4)
    ]

    with caplog.at_level("WARNING", logger="oxr.pipeline.reading_order"):
        result = asyncio.run(
            resolve_reading_orders(
                [blocks],
                [image],
                asyncio.Semaphore(1),
                client=Client(),
                config=config,
            )
        )

    assert result == [[0, 2, 3, 1]]
    assert len(calls) == 2
    assert "reading_order_supplemented" in caplog.text
    assert "missing_ids=[0, 2]" in caplog.text
