import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oxr.config.settings import settings
from oxr.pipeline import orchestrator
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.json_postprocess import serialize_final_blocks
from oxr.pipeline.postprocess.word_boxes import estimate_word_boxes, tokenize_line
from oxr.server.app import app
from oxr.server import routes


@pytest.fixture(autouse=True)
def configure_server_model_urls(monkeypatch):
    monkeypatch.setattr(settings.layout_analyze_model, "url", "http://layout:8081")
    monkeypatch.setattr(settings.oxr_model, "url", "http://oxr:8080/v1")


def test_tokenize_splits_latin_words_and_cjk_glyphs():
    tokens = [text for text, _start, _end in tokenize_line("Hello 世界, OK")]
    assert tokens == ["Hello", "世", "界", ",", "OK"]


def test_latin_word_boxes_are_left_to_right_inside_the_region():
    words = estimate_word_boxes("Hello world", [0, 10, 100, 20])
    assert [word["text"] for word in words] == ["Hello", "world"]
    first, second = words
    assert first["bbox"][0] == 0
    assert first["bbox"][2] < second["bbox"][0]
    assert second["bbox"][2] == 100
    assert first["bbox"][1] == second["bbox"][1] == 10
    assert first["bbox"][3] == second["bbox"][3] == 20


def test_multiline_word_boxes_stack_top_to_bottom():
    words = estimate_word_boxes("Top\nBottom", [0, 0, 40, 40])
    assert [word["text"] for word in words] == ["Top", "Bottom"]
    assert words[0]["bbox"][3] == words[1]["bbox"][1] == 20
    assert words[0]["bbox"][1] == 0
    assert words[1]["bbox"][3] == 40


def test_tall_single_line_region_uses_vertical_layout():
    words = estimate_word_boxes("你好", [0, 0, 10, 40])
    assert [word["text"] for word in words] == ["你", "好"]
    assert words[0]["bbox"] == [0, 0, 10, 20]
    assert words[1]["bbox"] == [0, 20, 10, 40]


def test_empty_or_degenerate_regions_yield_no_words():
    assert estimate_word_boxes("   ", [0, 0, 10, 10]) == []
    assert estimate_word_boxes("Hello", [0, 0, 0, 10]) == []
    assert estimate_word_boxes("Hello", [0, 0, 10]) == []


def test_serialize_omits_words_by_default_and_skips_non_text():
    text = Block(0, BlockType.TEXT, [0, 0, 40, 10], "Hello world", ContentFormat.TEXT)
    table = Block(1, BlockType.TABLE, [0, 10, 40, 20], "<table></table>", ContentFormat.HTML)
    records, notes = serialize_final_blocks([text, table])
    assert notes == []
    assert "words" not in records[0]
    assert "words" not in records[1]

    records, _notes = serialize_final_blocks([text, table], word_level_bboxes=True)
    assert [word["text"] for word in records[0]["words"]] == ["Hello", "world"]
    assert "words" not in records[1]


def test_process_document_adds_word_boxes_when_enabled(monkeypatch):
    async def analyze_hello(_images, _model_options=None):
        return [[Block(0, BlockType.TEXT, [0, 0, 100, 10], "Hello world", ContentFormat.TEXT)]]

    async def passthrough_pages(pages_blocks, *_args, **_kwargs):
        return pages_blocks

    monkeypatch.setattr(
        orchestrator, "pdf_to_images", lambda _file_bytes: [np.zeros((16, 16, 3), dtype=np.uint8)],
    )
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_hello)
    monkeypatch.setattr(orchestrator, "recognize_pages", passthrough_pages)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)

    default = asyncio.run(orchestrator.process_document(b"pdf", "demo.pdf", ["json"]))
    enabled = asyncio.run(
        orchestrator.process_document(
            b"pdf", "demo.pdf", ["json"], word_level_bboxes=True,
        )
    )

    default_block = default["outputs"]["json"]["pages"][0]["blocks"][0]
    enabled_block = enabled["outputs"]["json"]["pages"][0]["blocks"][0]
    assert "words" not in default_block
    assert [word["text"] for word in enabled_block["words"]] == ["Hello", "world"]
    assert enabled_block["bbox"] == [0, 0, 100, 10]


def test_parse_sync_passes_word_level_bboxes(monkeypatch):
    captured = {}

    async def fake_process_document(
        _file_bytes,
        _file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
        word_level_bboxes=False,
    ):
        captured["word_level_bboxes"] = word_level_bboxes
        captured["keep_header_footer"] = keep_header_footer
        return {
            "file_name": "demo.png",
            "outputs": {"json": {"pages": [{"blocks": [{"type": "Text", "words": []}]}]}},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "output_formats": "json",
                "feature_config": json.dumps({"word_level_bboxes": True}),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["warnings"] == []
    assert captured == {"word_level_bboxes": True, "keep_header_footer": False}


def test_parse_sync_rejects_non_boolean_word_level_bboxes():
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "feature_config": json.dumps({"word_level_bboxes": "yes"}),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 1133
    assert "feature_config.word_level_bboxes must be a boolean" in response.json()["message"]


def test_parse_async_forwards_word_level_bboxes(monkeypatch):
    captured = {}

    async def fake_process_document(
        _file_bytes,
        file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
        word_level_bboxes=False,
    ):
        captured["word_level_bboxes"] = word_level_bboxes
        return {
            "file_name": file_name,
            "outputs": {"json": {"pages": []}},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        submit = client.post(
            "/v1/parse/async",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "output_formats": "json",
                "feature_config": json.dumps({"word_level_bboxes": True}),
            },
        )
        task_id = submit.json()["data"]["task_id"]
        result = client.post(
            "/v1/parse/async_check",
            data={"api_key": "dummy", "task_id": task_id},
        )

    assert submit.json()["code"] == 0
    assert result.json()["data"]["status"] == "success"
    assert captured["word_level_bboxes"] is True
