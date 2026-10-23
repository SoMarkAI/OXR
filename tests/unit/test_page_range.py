import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oxr.config.settings import settings
from oxr.pipeline import orchestrator
from oxr.pipeline.errors import PipelineError
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.server.app import app
from oxr.server import routes
from oxr.utils.page_range import PAGE_RANGE_SYNTAX, PageRangeError, parse_page_range, resolve_page_indices


@pytest.fixture(autouse=True)
def configure_server_model_urls(monkeypatch):
    monkeypatch.setattr(settings.layout_analyze_model, "url", "http://layout:8081")
    monkeypatch.setattr(settings.oxr_model, "url", "http://oxr:8080/v1")


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("1", [0]),
        ("1-3,5,8-10", [0, 1, 2, 4, 7, 8, 9]),
        (" 3 , 1-2 ", [0, 1, 2]),
        ("2-2", [1]),
    ],
)
def test_parse_page_range_accepts_one_based_lists_and_spans(spec, expected):
    assert parse_page_range(spec) == expected


@pytest.mark.parametrize(
    "spec, message",
    [
        ("", "page_range must not be empty."),
        ("   ", "page_range must not be empty."),
        ("0", "greater than or equal to 1"),
        ("3-1", "inverted"),
        ("1-", PAGE_RANGE_SYNTAX),
        ("a,2", PAGE_RANGE_SYNTAX),
        ("1,,2", PAGE_RANGE_SYNTAX),
    ],
)
def test_parse_page_range_rejects_invalid_specs(spec, message):
    with pytest.raises(PageRangeError, match=message):
        parse_page_range(spec)


def test_resolve_page_indices_defaults_to_all_pages_and_rejects_overflow():
    assert resolve_page_indices(None, 3) == [0, 1, 2]
    assert resolve_page_indices([0, 2], 3) == [0, 2]
    with pytest.raises(PageRangeError, match="exceeds document length \\(3 pages\\)"):
        resolve_page_indices([3], 3)


def test_process_document_keeps_original_page_numbers_for_a_subset(monkeypatch):
    captured = {}

    def render_selected(_file_bytes, dpi=150, *, page_indices=None):
        captured["page_indices"] = list(page_indices)
        return [np.zeros((8, 8, 3), dtype=np.uint8) for _ in page_indices], 5

    async def analyze_layouts(images, _model_options=None):
        return [
            [Block(0, BlockType.TEXT, [0, 0, 8, 8], f"page-{index}", ContentFormat.TEXT)]
            for index in range(len(images))
        ]

    async def recognize_pages(pages_blocks, *_args, **_kwargs):
        return pages_blocks

    monkeypatch.setattr(orchestrator, "render_pdf_pages", render_selected)
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize_pages)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)

    result = asyncio.run(
        orchestrator.process_document(
            b"pdf", "demo.pdf", ["json", "markdown"], page_range=[1, 3],
        )
    )

    pages = result["outputs"]["json"]["pages"]
    assert captured["page_indices"] == [1, 3]
    assert [page["page_num"] for page in pages] == [1, 3]
    assert [page["blocks"][0]["content"] for page in pages] == ["page-0", "page-1"]
    assert result["metadata"]["page_num"] == 2
    assert result["metadata"]["total_pages"] == 5
    assert result["metadata"]["selected_pages"] == [2, 4]
    assert "page-0" in result["outputs"]["markdown"]


def test_process_document_omits_selected_pages_when_parsing_the_full_file(monkeypatch):
    monkeypatch.setattr(
        orchestrator, "pdf_to_images", lambda _file_bytes: [np.zeros((8, 8, 3), dtype=np.uint8)],
    )

    async def analyze_layouts(_images, _model_options=None):
        return [[Block(0, BlockType.TEXT, [0, 0, 8, 8], "Body", ContentFormat.TEXT)]]

    async def recognize_pages(pages_blocks, *_args, **_kwargs):
        return pages_blocks

    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize_pages)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)

    result = asyncio.run(orchestrator.process_document(b"pdf", "demo.pdf", ["json"]))
    assert result["outputs"]["json"]["pages"][0]["page_num"] == 0
    assert result["metadata"]["page_num"] == 1
    assert result["metadata"]["total_pages"] == 1
    assert "selected_pages" not in result["metadata"]


def test_load_document_pages_rejects_image_page_range_other_than_first(monkeypatch):
    monkeypatch.setattr(
        orchestrator.cv2, "imdecode", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    with pytest.raises(PipelineError) as exc_info:
        orchestrator._load_document_pages(b"image", "scan.png", [1])
    assert exc_info.value.api_code == 1133
    assert "1 pages" in exc_info.value.public_message


def test_parse_sync_passes_page_range(monkeypatch):
    captured = {}

    async def fake_process_document(
        _file_bytes,
        _file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
        page_range=None,
    ):
        captured["page_range"] = page_range
        return {
            "file_name": "demo.pdf",
            "outputs": {"json": {"pages": []}},
            "metadata": {"page_num": 2, "total_pages": 10, "selected_pages": [2, 4]},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.pdf", b"fake-pdf", "application/pdf")},
            data={
                "api_key": "dummy",
                "output_formats": "json",
                "page_range": "2,4",
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert captured["page_range"] == [1, 3]
    assert response.json()["data"]["metadata"]["selected_pages"] == [2, 4]


def test_parse_sync_rejects_invalid_page_range_before_processing(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("Invalid page_range must not start processing")

    monkeypatch.setattr(routes, "process_document", unexpected)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.pdf", b"fake-pdf", "application/pdf")},
            data={"api_key": "dummy", "page_range": "3-1"},
        )

    assert response.status_code == 200
    assert response.json()["code"] == 1133
    assert "inverted" in response.json()["message"]


def test_parse_sync_reports_out_of_range_page_range_as_parameter_error(monkeypatch):
    async def fail_process_document(*_args, **_kwargs):
        raise PipelineError(
            "page_range exceeds document length (2 pages).",
            public_message="page_range exceeds document length (2 pages).",
            api_code=1133,
        )

    monkeypatch.setattr(routes, "process_document", fail_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.pdf", b"fake-pdf", "application/pdf")},
            data={"api_key": "dummy", "page_range": "1-5"},
        )

    assert response.status_code == 200
    assert response.json()["code"] == 1133
    assert response.json()["message"] == "page_range exceeds document length (2 pages)."


def test_parse_async_forwards_page_range(monkeypatch):
    captured = {}

    async def fake_process_document(
        _file_bytes,
        file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
        page_range=None,
    ):
        captured["page_range"] = page_range
        return {
            "file_name": file_name,
            "outputs": {"json": {"pages": []}},
            "metadata": {"page_num": 1, "total_pages": 8, "selected_pages": [8]},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        submit = client.post(
            "/v1/parse/async",
            files={"file": ("demo.pdf", b"fake-pdf", "application/pdf")},
            data={"api_key": "dummy", "output_formats": "json", "page_range": "8"},
        )
        task_id = submit.json()["data"]["task_id"]
        result = client.post(
            "/v1/parse/async_check",
            data={"api_key": "dummy", "task_id": task_id},
        )

    assert submit.json()["code"] == 0
    assert result.json()["data"]["status"] == "success"
    assert captured["page_range"] == [7]
    assert result.json()["data"]["metadata"]["selected_pages"] == [8]
