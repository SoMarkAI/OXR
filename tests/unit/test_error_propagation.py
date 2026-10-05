import asyncio

import numpy as np
import pytest
from fastapi.testclient import TestClient

from oxr.config.settings import settings
from oxr.pipeline.errors import PipelineError
from oxr.pipeline.layout import analyze_layouts
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.recognition import recognize_block
from oxr.server.app import app
from oxr.server import routes


INVALID_DOCUMENT_MESSAGE = "Unable to decode document into valid page images."


@pytest.fixture(autouse=True)
def configure_server_model_urls(monkeypatch):
    monkeypatch.setattr(settings.layout_analyze_model, "url", "http://layout:8081")
    monkeypatch.setattr(settings.oxr_model, "url", "http://oxr:8080/v1")


class UpstreamServiceUnavailableError(Exception):
    status_code = 503


def test_layout_failure_raises_pipeline_error(monkeypatch):
    async def fail_layout(_encoded_images):
        raise UpstreamServiceUnavailableError("Error code: 503")

    monkeypatch.setattr("oxr.pipeline.layout.layout_client.request_batch", fail_layout)
    monkeypatch.setattr(settings.layout_analyze_model, "retry_times", 1)

    with np.testing.assert_raises(PipelineError) as exc_info:
        asyncio.run(analyze_layouts([np.zeros((8, 8, 3), dtype=np.uint8)]))

    assert exc_info.exception.status_code == 503
    assert "Layout analysis failed" in str(exc_info.exception)


def test_layout_timeout_has_clear_error_message(monkeypatch):
    async def fail_layout(_encoded_images):
        raise TimeoutError()

    monkeypatch.setattr("oxr.pipeline.layout.layout_client.request_batch", fail_layout)
    monkeypatch.setattr(settings.layout_analyze_model, "retry_times", 1)
    monkeypatch.setattr(settings.layout_analyze_model, "timeout", 7)

    with np.testing.assert_raises(PipelineError) as exc_info:
        asyncio.run(analyze_layouts([np.zeros((8, 8, 3), dtype=np.uint8)]))

    assert "Timeout after 7s" in str(exc_info.exception)


def test_recognition_failure_raises_pipeline_error(monkeypatch):
    async def fail_recognize(_image, _model_options=None):
        raise UpstreamServiceUnavailableError("Error code: 503")

    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_text", fail_recognize)
    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)

    block = Block(
        idx=0,
        type=BlockType.TEXT,
        bbox=[0, 0, 8, 8],
        content="",
        format=ContentFormat.TEXT,
    )

    with np.testing.assert_raises(PipelineError) as exc_info:
        asyncio.run(
            recognize_block(
                block,
                np.zeros((8, 8, 3), dtype=np.uint8),
                asyncio.Semaphore(1),
                "demo.png",
                0,
            )
        )

    assert exc_info.exception.status_code == 503
    assert "Recognition failed for block 0" in str(exc_info.exception)


def test_parse_sync_returns_somark_error_envelope(monkeypatch):
    async def fail_process_document(
        _file_bytes,
        _file_name,
        _output_formats,
        keep_header_footer=False,
        model_options=None,
        element_formats=None,
    ):
        raise PipelineError("Layout analysis failed after 1 attempts: Error code: 503", status_code=503)

    monkeypatch.setattr(routes, "process_document", fail_process_document)

    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={"api_key": "dummy"},
        )

    assert response.status_code == 200
    assert response.json()["code"] == 1137
    assert response.json()["message"] == "Unable to process this document. Please try again later."
    assert response.json()["warnings"] == []


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("error_type", [RuntimeError, PipelineError, OSError])
def test_parse_failures_hide_backend_details(monkeypatch, caplog, mode, error_type):
    details = ("/synthetic-private/checkpoint.pth", "http://backend.example.invalid", "DUMMY_CREDENTIAL")

    async def fail_process_document(*args, **kwargs):
        raise error_type(" ".join(details))

    monkeypatch.setattr(routes, "process_document", fail_process_document)
    with TestClient(app) as client:
        response = client.post(
            f"/v1/parse/{mode}",
            files={"file": ("demo.png", b"fake-image", "image/png")},
            data={"api_key": "dummy"},
        )
        assert response.status_code == 200
        if mode == "async":
            assert response.json()["code"] == 0
            response = client.post(
                "/v1/parse/async_check",
                data={"api_key": "dummy", "task_id": response.json()["data"]["task_id"]},
            )
            assert response.status_code == 200
            assert response.json()["code"] == 0
            assert response.json()["data"]["status"] == "failed"
            message = response.json()["data"]["error"]
        else:
            assert response.json()["code"] == 1137
            assert response.json()["warnings"] == []
            assert response.json()["data"] is None
            message = response.json()["message"]

    assert message == "Unable to process this document. Please try again later."
    assert "Document processing failed" in caplog.text
    assert error_type.__name__ in caplog.text
    assert all(detail not in response.text and detail not in caplog.text for detail in details)


def test_decoder_exception_keeps_safe_public_message(monkeypatch):
    from oxr.pipeline import orchestrator

    def fail_pdf_decode(*args, **kwargs):
        raise RuntimeError("/synthetic-private/checkpoint.pth DUMMY_CREDENTIAL")

    monkeypatch.setattr(orchestrator, "pdf_to_images", fail_pdf_decode)
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("broken.pdf", b"fake-pdf", "application/pdf")},
            data={"api_key": "dummy"},
        )

    assert response.status_code == 200
    assert response.json()["code"] == 1137
    assert response.json()["message"] == INVALID_DOCUMENT_MESSAGE
    assert "synthetic-private" not in response.text and "DUMMY_CREDENTIAL" not in response.text


@pytest.mark.parametrize("file_bytes", [b"", b"not-an-image"])
def test_parse_sync_returns_stable_error_for_invalid_image_bytes(file_bytes):
    with TestClient(app) as client:
        response = client.post(
            "/v1/parse/sync",
            files={"file": ("broken.png", file_bytes, "image/png")},
            data={"api_key": "dummy"},
        )

    assert response.status_code == 200
    assert response.json()["code"] == 1137
    assert response.json()["message"] == INVALID_DOCUMENT_MESSAGE
    assert "OpenCV" not in response.json()["message"]


@pytest.mark.parametrize("file_bytes", [b"", b"not-an-image"])
def test_parse_async_records_stable_error_for_invalid_image_bytes(file_bytes):
    with TestClient(app) as client:
        submit = client.post(
            "/v1/parse/async",
            files={"file": ("broken.png", file_bytes, "image/png")},
            data={"api_key": "dummy"},
        )
        task_id = submit.json()["data"]["task_id"]
        response = client.post(
            "/v1/parse/async_check",
            data={"api_key": "dummy", "task_id": task_id},
        )

    assert submit.json()["code"] == 0
    assert response.json()["data"]["status"] == "failed"
    assert response.json()["data"]["error"] == INVALID_DOCUMENT_MESSAGE
    assert "OpenCV" not in response.json()["data"]["error"]
