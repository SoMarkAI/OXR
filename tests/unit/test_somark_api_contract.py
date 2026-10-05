import asyncio
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from oxr.cli.main import app as cli_app
from oxr.config.settings import settings
from oxr.pipeline import orchestrator
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.server.app import app
from oxr.server import routes


@pytest.fixture(autouse=True)
def configure_server_model_urls(monkeypatch):
    monkeypatch.setattr(settings.layout_analyze_model, "url", "http://layout:8081")
    monkeypatch.setattr(settings.oxr_model, "url", "http://oxr:8080/v1")


def test_oxr_help_exposes_only_supported_commands():
    assert {command.name for command in cli_app.registered_commands} == {"serve"}

    result = CliRunner().invoke(cli_app, ["--help"])

    assert result.exit_code == 0
    assert "serve" in result.output
    assert "app" not in result.output
    assert "parse" not in result.output


def test_removed_app_command_is_rejected(monkeypatch):
    def unexpected_server_start(*args, **kwargs):
        pytest.fail("An unsupported command must not start the server")

    monkeypatch.setattr("oxr.cli.cmd_serve.uvicorn.run", unexpected_server_start)
    result = CliRunner().invoke(cli_app, ["app"])

    assert result.exit_code == 2
    assert "No such command 'app'" in result.output


def test_parse_sync_accepts_default_somark_element_formats(monkeypatch):
    captured = {}

    async def fake_process_document(
        _file_bytes,
        _file_name,
        output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        captured["output_formats"] = output_formats
        captured["keep_header_footer"] = keep_header_footer
        captured["model_options"] = model_options
        captured["element_formats"] = element_formats
        return {
            "file_name": "demo.png",
            "outputs": {"markdown": "ok", "json": {"pages": []}},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "element_formats": json.dumps({
                    "image": "url",
                    "formula": "latex",
                    "table": "html",
                    "cs": "image",
                }),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["message"] == "任务成功"
    assert response.json()["warnings"] == []
    assert captured["element_formats"] == {
        "image": "url",
        "formula": "latex",
        "table": "html",
        "cs": "image",
    }


def test_parse_sync_reports_multiple_parameter_errors():
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "output_formats": "zip",
                "element_formats": json.dumps({
                    "image": "file",
                    "formula": "mathml",
                    "table": "image",
                }),
            },
        )

    message = response.json()["message"]
    assert response.status_code == 200
    assert response.json()["code"] == 1133
    assert response.json()["warnings"] == []
    assert "output_formats=zip" in message
    assert "element_formats.image=file" in message
    assert "element_formats.formula=mathml" in message
    assert "element_formats.table=image" in message


def test_parse_sync_rejects_zip_output_format():
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "output_formats": "zip",
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 1133
    assert response.json()["warnings"] == []
    assert "output_formats=zip" in response.json()["message"]


def test_parse_sync_passes_feature_config_keep_header_footer(monkeypatch):
    captured = {}

    async def fake_process_document(
        _file_bytes,
        _file_name,
        output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        captured["output_formats"] = output_formats
        captured["keep_header_footer"] = keep_header_footer
        captured["model_options"] = model_options
        captured["element_formats"] = element_formats
        return {
            "file_name": "demo.png",
            "outputs": {"markdown": "ok", "json": {"pages": []}},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "output_formats": ["markdown", "json"],
                "feature_config": json.dumps({"keep_header_footer": True}),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["warnings"] == []
    assert captured == {
        "output_formats": ["markdown", "json"],
        "keep_header_footer": True,
        "model_options": {},
        "element_formats": {
            "image": "url",
            "formula": "latex",
            "table": "html",
            "cs": "smiles",
        },
    }


def test_parse_sync_passes_extra_model_options(monkeypatch):
    captured = {}

    async def fake_process_document(
        _file_bytes,
        _file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        captured["keep_header_footer"] = keep_header_footer
        captured["model_options"] = model_options
        captured["element_formats"] = element_formats
        return {
            "file_name": "demo.png",
            "outputs": {"markdown": "ok", "json": {"pages": []}},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "feature_config": json.dumps({"keep_header_footer": True}),
                "extra": json.dumps({
                    "temperature": 0.2,
                    "top_p": 0.9,
                    "repetition_penalty": 1.1,
                }),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["warnings"] == []
    assert captured == {
        "keep_header_footer": True,
        "model_options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
        },
        "element_formats": {
            "image": "url",
            "formula": "latex",
            "table": "html",
            "cs": "smiles",
        },
    }


def test_parse_sync_warns_for_unsupported_feature_config_true(monkeypatch):
    async def fake_process_document(
        _file_bytes,
        _file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        return {
            "file_name": "demo.png",
            "outputs": {"markdown": "ok", "json": {"pages": []}},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "feature_config": json.dumps({
                    "enable_text_cross_page": True,
                    "enable_table_cross_page": True,
                }),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["message"] == "任务成功"
    assert len(response.json()["warnings"]) == 2
    assert any(
        "feature_config.enable_text_cross_page=true" in warning
        for warning in response.json()["warnings"]
    )
    assert any(
        "feature_config.enable_table_cross_page=true" in warning
        for warning in response.json()["warnings"]
    )


def test_parse_sync_ignores_unsupported_feature_config_false(monkeypatch):
    async def fake_process_document(
        _file_bytes,
        _file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        return {
            "file_name": "demo.png",
            "outputs": {"markdown": "ok", "json": {"pages": []}},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "feature_config": json.dumps({"enable_text_cross_page": False}),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["message"] == "任务成功"
    assert response.json()["warnings"] == []


def test_parse_async_warnings_keep_submit_message_stable(monkeypatch):
    async def fake_process_document(
        _file_bytes,
        file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        return {
            "file_name": file_name,
            "outputs": {"markdown": "ok"},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/async",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={
                "api_key": "dummy",
                "feature_config": json.dumps({"enable_text_cross_page": True}),
            },
        )

    assert response.status_code == 200
    assert response.json()["code"] == 0
    assert response.json()["message"] == "任务已提交"
    assert len(response.json()["warnings"]) == 1
    assert "feature_config.enable_text_cross_page=true" in response.json()["warnings"][0]


def test_parse_sync_rejects_invalid_extra_model_options():
    invalid_extras = [
        "{",
        "[]",
        json.dumps({"unsupported": True}),
        {"temperature": "hot"},
        {"temperature": 2.1},
        {"top_p": 1.1},
        {"repetition_penalty": 0},
    ]

    with TestClient(app) as client:
        for extra in invalid_extras:
            extra_value = extra if isinstance(extra, str) else json.dumps(extra)
            response = client.post(
                "/v1/parse/sync",
                files={"file": ("demo.png", b"fake-image", "image/png")},
                data={
                    "api_key": "dummy",
                    "extra": extra_value,
                },
            )

            assert response.status_code == 200
            assert response.json()["code"] == 1133
            assert response.json()["warnings"] == []
            assert "extra" in response.json()["message"]


def test_async_check_returns_somark_style_result(monkeypatch):
    async def fake_process_document(
        _file_bytes,
        file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        return {
            "file_name": file_name,
            "outputs": {"markdown": "ok"},
            "metadata": {"page_num": 1, "file_type": ".png"},
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        submit = client.post(
            "/v1/parse/async",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={"api_key": "dummy", "output_formats": "markdown"},
        )
        task_id = submit.json()["data"]["task_id"]

        response = client.post(
            "/v1/parse/async_check",
            data={"api_key": "dummy", "task_id": task_id},
        )

    data = response.json()["data"]
    assert response.status_code == 200
    assert response.json()["warnings"] == []
    assert data["task_id"] == task_id
    assert data["status"] == "success"
    assert data["file_name"] == "demo.png"
    assert data["result"]["outputs"]["markdown"] == "ok"


def test_async_check_preserves_debug_metadata(monkeypatch):
    async def fake_process_document(
        _file_bytes,
        file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        return {
            "file_name": file_name,
            "outputs": {"markdown": "ok"},
            "metadata": {
                "page_num": 1,
                "file_type": ".png",
                "processing_time_ms": 10,
                "debug_timing": {
                    "decode_time_ms": 1,
                    "layout_time_ms": 2,
                    "recognition_time_ms": 6,
                    "assembly_time_ms": 1,
                },
                "debug_stats": {
                    "total_blocks": 1,
                    "blocks_per_page": [1],
                    "block_type_counts": {BlockType.TEXT.value: 1},
                },
            },
        }

    monkeypatch.setattr(routes, "process_document", fake_process_document)

    with TestClient(app) as client:
        submit = client.post(
            "/v1/parse/async",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={"api_key": "dummy", "output_formats": "markdown"},
        )
        task_id = submit.json()["data"]["task_id"]

        response = client.post(
            "/v1/parse/async_check",
            data={"api_key": "dummy", "task_id": task_id},
        )

    metadata = response.json()["data"]["metadata"]
    assert metadata["debug_timing"]["layout_time_ms"] == 2
    assert metadata["debug_stats"]["total_blocks"] == 1


def test_process_document_filters_header_footer_by_default(monkeypatch):
    captured = {}

    async def fake_analyze_layouts(_images, _model_options=None):
        return [[
            Block(0, BlockType.PAGE_HEADER, [0, 0, 10, 5], "Header", ContentFormat.TEXT),
            Block(1, BlockType.TEXT, [0, 5, 10, 10], "Body", ContentFormat.TEXT),
            Block(2, BlockType.PAGE_FOOTER, [0, 10, 10, 15], "Footer", ContentFormat.TEXT),
        ]]

    async def fake_recognize_pages(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        captured["recognized_types"] = [
            block.type
            for page_blocks in pages_blocks
            for block in page_blocks
        ]
        return pages_blocks

    monkeypatch.setattr(orchestrator, "pdf_to_images", lambda _file_bytes: [np.zeros((16, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(orchestrator, "analyze_layouts", fake_analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", fake_recognize_pages)

    result = asyncio.run(orchestrator.process_document(b"pdf", "demo.pdf", ["markdown", "json"]))

    block_types = [block["type"] for block in result["outputs"]["json"]["pages"][0]["blocks"]]
    assert block_types == [BlockType.TEXT.value]
    assert captured["recognized_types"] == [BlockType.TEXT]
    assert "Body" in result["outputs"]["markdown"]
    assert "Header" not in result["outputs"]["markdown"]
    assert "Footer" not in result["outputs"]["markdown"]


def test_process_document_keeps_header_footer_when_enabled(monkeypatch):
    async def fake_analyze_layouts(_images, _model_options=None):
        return [[
            Block(0, BlockType.PAGE_HEADER, [0, 0, 10, 5], "Header", ContentFormat.TEXT),
            Block(1, BlockType.TEXT, [0, 5, 10, 10], "Body", ContentFormat.TEXT),
            Block(2, BlockType.PAGE_FOOTER, [0, 10, 10, 15], "Footer", ContentFormat.TEXT),
        ]]

    async def fake_recognize_pages(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        return pages_blocks

    monkeypatch.setattr(orchestrator, "pdf_to_images", lambda _file_bytes: [np.zeros((16, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(orchestrator, "analyze_layouts", fake_analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", fake_recognize_pages)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)

    result = asyncio.run(
        orchestrator.process_document(b"pdf", "demo.pdf", ["markdown", "json"], keep_header_footer=True)
    )

    block_types = [block["type"] for block in result["outputs"]["json"]["pages"][0]["blocks"]]
    assert block_types == [BlockType.PAGE_HEADER.value, BlockType.TEXT.value, BlockType.PAGE_FOOTER.value]
    assert "Header" in result["outputs"]["markdown"]
    assert "Footer" in result["outputs"]["markdown"]


def test_process_document_passes_model_options(monkeypatch):
    captured = {}
    model_options = {"temperature": 0.3, "top_p": 0.8, "repetition_penalty": 1.2}

    async def fake_analyze_layouts(_images, options=None):
        captured["layout_options"] = options
        return [[Block(0, BlockType.TEXT, [0, 0, 10, 10], "Body", ContentFormat.TEXT)]]

    async def fake_recognize_pages(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        options=None,
        element_formats=None,
    ):
        captured["recognition_options"] = options
        captured["element_formats"] = element_formats
        return pages_blocks

    monkeypatch.setattr(orchestrator, "pdf_to_images", lambda _file_bytes: [np.zeros((16, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(orchestrator, "analyze_layouts", fake_analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", fake_recognize_pages)

    asyncio.run(
        orchestrator.process_document(
            b"pdf",
            "demo.pdf",
            ["markdown", "json"],
            model_options=model_options,
        )
    )

    assert captured == {
        "layout_options": model_options,
        "recognition_options": model_options,
        "element_formats": None,
    }


def test_process_document_hides_debug_metadata_by_default(monkeypatch):
    async def fake_analyze_layouts(_images, _model_options=None):
        return [[Block(0, BlockType.TEXT, [0, 0, 10, 10], "Body", ContentFormat.TEXT)]]

    async def fake_recognize_pages(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        return pages_blocks

    monkeypatch.setattr(orchestrator.settings, "debug", False)
    monkeypatch.setattr(orchestrator, "pdf_to_images", lambda _file_bytes: [np.zeros((16, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(orchestrator, "analyze_layouts", fake_analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", fake_recognize_pages)

    result = asyncio.run(orchestrator.process_document(b"pdf", "demo.pdf", ["markdown", "json"]))

    assert "processing_time_ms" in result["metadata"]
    assert "debug_timing" not in result["metadata"]
    assert "debug_stats" not in result["metadata"]


def test_process_document_returns_debug_metadata_when_enabled(monkeypatch):
    async def fake_analyze_layouts(_images, _model_options=None):
        return [[
            Block(0, BlockType.TEXT, [0, 0, 10, 10], "Body", ContentFormat.TEXT),
            Block(1, BlockType.TABLE, [0, 10, 10, 20], "<table></table>", ContentFormat.HTML),
        ]]

    async def fake_recognize_pages(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        return pages_blocks

    monkeypatch.setattr(orchestrator.settings, "debug", True)
    monkeypatch.setattr(orchestrator, "pdf_to_images", lambda _file_bytes: [np.zeros((16, 16, 3), dtype=np.uint8)])
    monkeypatch.setattr(orchestrator, "analyze_layouts", fake_analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", fake_recognize_pages)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)

    result = asyncio.run(orchestrator.process_document(b"pdf", "demo.pdf", ["markdown", "json"]))

    timing = result["metadata"]["debug_timing"]
    stats = result["metadata"]["debug_stats"]
    assert set(timing) == {
        "decode_time_ms",
        "layout_time_ms",
        "recognition_time_ms",
        "assembly_time_ms",
    }
    assert stats == {
        "total_blocks": 2,
        "blocks_per_page": [2],
        "block_type_counts": {
            BlockType.TABLE.value: 1,
            BlockType.TEXT.value: 1,
        },
    }
