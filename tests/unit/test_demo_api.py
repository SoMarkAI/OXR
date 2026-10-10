import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from oxr.config.settings import settings
from oxr.pipeline import orchestrator
from oxr.pipeline.errors import PipelineError
from oxr.server import demo_api
from oxr.server.app import app
from oxr.server.demo_store import DemoStore


@pytest.fixture
def demo_store(monkeypatch, tmp_path):
    store = DemoStore(tmp_path / "jobs")
    monkeypatch.setattr(demo_api, "DemoStore", lambda: store)
    monkeypatch.setattr(settings.layout_analyze_model, "url", "http://layout:8081")
    monkeypatch.setattr(settings.oxr_model, "url", "http://oxr:8080/v1")
    return store


def image_bytes():
    success, encoded = cv2.imencode(".png", np.full((18, 24, 3), 255, dtype=np.uint8))
    assert success
    return encoded.tobytes()


def wait_status(client, job_id, expected):
    for _ in range(200):
        response = client.get(f"/v1/demo/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()["data"]
        if data["status"] == expected:
            return data
        time.sleep(0.01)
    pytest.fail(f"Expected {expected}; received {data}")


def test_demo_upload_publishes_page_before_result_and_decodes_only_once(monkeypatch, demo_store):
    release = threading.Event()
    decode_calls = []
    original_decode = orchestrator._decode_document_images

    def decode(data, name):
        decode_calls.append(name)
        return original_decode(data, name)

    async def process(images, name, formats, **kwargs):
        assert len(images) == 1 and images[0].shape == (18, 24, 3)
        assert formats == ["markdown", "json"]
        assert kwargs["demo"] is True
        assert kwargs["keep_header_footer"] is True
        assert kwargs["started_at"] > 0
        while not release.is_set():
            await asyncio.sleep(0.01)
        (kwargs["asset_dir"] / "block-test.png").write_bytes(image_bytes())
        return {
            "file_name": name,
            "outputs": {"markdown": "# Example", "json": {"pages": []}},
            "metadata": {"page_num": 1},
            "demo_pages": [{"page_num": 0, "markdown": "# Example", "blocks": []}],
        }

    monkeypatch.setattr(orchestrator, "_decode_document_images", decode)
    monkeypatch.setattr(orchestrator, "process_decoded_document", process)
    with TestClient(app) as client:
        response = client.post("/v1/demo/jobs", files={"file": ("example.png", image_bytes(), "image/png")})
        assert response.status_code == 200
        job_id = response.json()["data"]["id"]
        state = wait_status(client, job_id, "processing")
        assert state["pages"][0]["page_size"] == {"w": 24, "h": 18}
        assert "assets" not in state and "expires_at" not in state and "outputs" not in state
        page = client.get(state["pages"][0]["image_url"])
        assert page.status_code == 200 and page.headers["content-type"] == "image/png"
        assert client.get(f"/v1/demo/jobs/{job_id}/result").status_code == 409
        release.set()
        state = wait_status(client, job_id, "success")
        result = client.get(state["result_url"]).json()["data"]
        assert result["outputs"]["markdown"] == "# Example"
        assert result["demo_pages"][0]["page_num"] == 0
        assert client.get(f"/v1/demo/jobs/{job_id}/assets/block-test.png").status_code == 200
        assert decode_calls == ["example.png"]


def test_demo_pdf_pages_use_decoder_dimensions(monkeypatch, demo_store):
    called = []

    def decode(data, name):
        called.append((data, name))
        return [np.zeros((20, 30, 3), dtype=np.uint8), np.zeros((40, 15, 3), dtype=np.uint8)]

    async def process(images, name, formats, **kwargs):
        return {"file_name": name, "outputs": {}, "metadata": {}, "demo_pages": []}

    monkeypatch.setattr(orchestrator, "_decode_document_images", decode)
    monkeypatch.setattr(orchestrator, "process_decoded_document", process)
    with TestClient(app) as client:
        response = client.post("/v1/demo/jobs", files={"file": ("two.pdf", b"pdf")})
        state = wait_status(client, response.json()["data"]["id"], "success")
        assert called == [(b"pdf", "two.pdf")]
        assert [page["page_size"] for page in state["pages"]] == [{"w": 30, "h": 20}, {"w": 15, "h": 40}]
        assert [page["page_num"] for page in state["pages"]] == [0, 1]


@pytest.mark.parametrize("submitted, expected", [
    ({}, {"temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0}),
    ({"temperature": "0.4", "top_p": "0.8", "repetition_penalty": "1.3"},
     {"temperature": 0.4, "top_p": 0.8, "repetition_penalty": 1.3}),
    ({"temperature": "0", "top_p": "0", "repetition_penalty": "0.1"},
     {"temperature": 0.0, "top_p": 0.0, "repetition_penalty": 0.1}),
    ({"temperature": "2", "top_p": "1", "repetition_penalty": "2"},
     {"temperature": 2.0, "top_p": 1.0, "repetition_penalty": 2.0}),
    ({"top_p": "0.5"}, {"temperature": 0.0, "top_p": 0.5, "repetition_penalty": 1.0}),
])
def test_demo_model_options_reach_pipeline(monkeypatch, demo_store, submitted, expected):
    received = []

    async def process(images, name, formats, **kwargs):
        received.append(kwargs["model_options"])
        return {"file_name": name, "outputs": {}, "metadata": {}, "demo_pages": []}

    monkeypatch.setattr(orchestrator, "process_decoded_document", process)
    with TestClient(app) as client:
        response = client.post(
            "/v1/demo/jobs", data=submitted,
            files={"file": ("example.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        wait_status(client, response.json()["data"]["id"], "success")
    assert received == [expected]


@pytest.mark.parametrize("submitted, expected", [({}, True),
    ({"keep_header_footer": "true"}, True), ({"keep_header_footer": "false"}, False),
])
def test_demo_header_footer_option_reaches_pipeline_separately(
    monkeypatch, demo_store, submitted, expected,
):
    received = []

    async def process(images, name, formats, **kwargs):
        received.append(kwargs)
        return {"file_name": name, "outputs": {}, "metadata": {}, "demo_pages": []}

    monkeypatch.setattr(orchestrator, "process_decoded_document", process)
    with TestClient(app) as client:
        response = client.post(
            "/v1/demo/jobs", data=submitted,
            files={"file": ("example.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 200
        wait_status(client, response.json()["data"]["id"], "success")
    assert len(received) == 1
    assert received[0]["keep_header_footer"] is expected
    assert "keep_header_footer" not in received[0]["model_options"]


def test_demo_rejects_invalid_header_footer_option_before_creating_job(monkeypatch, demo_store):
    def unexpected_create(*_args):
        pytest.fail("Invalid header/footer option must not create a job")

    monkeypatch.setattr(demo_store, "create", unexpected_create)
    with TestClient(app) as client:
        response = client.post(
            "/v1/demo/jobs", data={"keep_header_footer": "sometimes"},
            files={"file": ("example.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 422
        assert "keep_header_footer" in response.text
        assert not app.state.demo_tasks
    assert not list(demo_store.root.glob("*/manifest.json"))


@pytest.mark.parametrize("name, value", [
    (name, value)
    for name in ("temperature", "top_p", "repetition_penalty")
    for value in ("", " ", "hot", "NaN", "Infinity", "-Infinity")
] + [
    ("temperature", "-0.1"), ("temperature", "2.1"),
    ("top_p", "-0.1"), ("top_p", "1.1"),
    ("repetition_penalty", "0"), ("repetition_penalty", "0.09"),
    ("repetition_penalty", "2.1"),
])
def test_demo_rejects_invalid_model_options_before_creating_job(monkeypatch, demo_store, name, value):
    def unexpected_create(*_args):
        pytest.fail("Invalid model parameters must not create a job")

    monkeypatch.setattr(demo_store, "create", unexpected_create)
    with TestClient(app) as client:
        response = client.post(
            "/v1/demo/jobs", data={name: value},
            files={"file": ("example.png", image_bytes(), "image/png")},
        )
        assert response.status_code == 422
        body = response.json()
        assert body["code"] == 422
        assert name in body["message"] and "between" in body["message"]
        assert body["data"] is None
        assert not app.state.demo_tasks
    assert not list(demo_store.root.glob("*/manifest.json"))


def test_upload_errors_and_decode_failure(monkeypatch, demo_store):
    with TestClient(app) as client:
        assert client.post("/v1/demo/jobs", files={"file": ("bad.txt", b"data")}).status_code == 415
        assert client.post("/v1/demo/jobs", files={"file": ("empty.png", b"")}).status_code == 422
        monkeypatch.setattr(demo_api, "MAX_UPLOAD_BYTES", 2)
        assert client.post("/v1/demo/jobs", files={"file": ("big.png", b"123")}).status_code == 413
        monkeypatch.setattr(demo_api, "MAX_UPLOAD_BYTES", 100)
        response = client.post("/v1/demo/jobs", files={"file": ("bad.png", b"not an image")})
        state = wait_status(client, response.json()["data"]["id"], "failed")
        assert "Unable to decode" in state["error"]
        assert state["result_url"] is None
        assert client.get("/v1/demo/jobs/unknown").status_code == 404


def test_assets_are_job_scoped_allowlisted_and_expire(demo_store):
    first = demo_store.create("same.png")["id"]
    second = demo_store.create("same.png")["id"]
    directory = demo_store.job_dir(first) / "images"
    (directory / "page-0000.png").write_bytes(image_bytes())
    (directory / "secret.png").write_bytes(b"not registered")
    demo_store.update(first, assets=["page-0000.png"])
    with TestClient(app) as client:
        assert client.get(f"/v1/demo/jobs/{first}/assets/page-0000.png").status_code == 200
        assert client.get(f"/v1/demo/jobs/{second}/assets/page-0000.png").status_code == 404
        assert client.get(f"/v1/demo/jobs/{first}/assets/secret.png").status_code == 404
        assert client.get(f"/v1/demo/jobs/{first}/assets/manifest.json").status_code == 404
        manifest_path = demo_store.job_dir(first) / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["expires_at"] = 0
        manifest_path.write_text(json.dumps(manifest))
        os.utime(demo_store.job_dir(first) / "lease", (0, 0))
        assert client.get(f"/v1/demo/jobs/{first}").status_code == 404
        assert client.get(f"/v1/demo/jobs/{first}/assets/page-0000.png").status_code == 404
        demo_store.cleanup()
        assert not demo_store.job_dir(first).exists()


def test_demo_serves_bundled_ui_and_keeps_parse_routes(demo_store):
    with TestClient(app) as client:
        for url in ("/demo", "/demo/"):
            response = client.get(url)
            assert response.status_code == 200
            assert "<title>OXR Online Demo</title>" in response.text
        response = client.get("/demo/assets/style/index.css")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/css")
        paths = client.get("/openapi.json").json()["paths"]
        assert {"/v1/parse/sync", "/v1/parse/async", "/v1/parse/async_check"} <= paths.keys()
        assert paths["/v1/demo/jobs"]["post"]["tags"] == ["OXR Online Demo"]


def test_asset_symlink_cannot_escape_job_directory(demo_store, tmp_path):
    job_id = demo_store.create("a.png")["id"]
    secret = tmp_path / "outside.png"
    secret.write_bytes(b"private")
    link = demo_store.job_dir(job_id) / "images" / "linked.png"
    try:
        link.symlink_to(secret)
    except OSError:
        pytest.skip("Symlinks are unavailable on this platform.")
    demo_store.update(job_id, assets=["linked.png"])
    with pytest.raises(KeyError):
        demo_store.asset(job_id, "linked.png")
    demo_store.publish_result(job_id, {"metadata": {}})
    assert "linked.png" not in demo_store.get(job_id)["assets"]


def test_storage_exhaustion_marks_failed_and_removes_page_payload(monkeypatch, demo_store):
    original_check = demo_store.ensure_capacity
    checks = []

    def check(*args):
        checks.append(True)
        if len(checks) > 1:
            raise OSError("Temporary storage is full.")
        return original_check(*args)

    monkeypatch.setattr(demo_store, "ensure_capacity", check)
    with TestClient(app) as client:
        response = client.post("/v1/demo/jobs", files={"file": ("a.png", image_bytes())})
        job_id = response.json()["data"]["id"]
        state = wait_status(client, job_id, "failed")
        assert state["error"] == "Temporary storage is unavailable or full. Try again later."
        assert state["pages"] == []
        assert not (demo_store.job_dir(job_id) / "images").exists()


@pytest.mark.parametrize("stage", ["decode", "recognition", "publish"])
@pytest.mark.parametrize("error_type", [RuntimeError, PipelineError, OSError])
def test_demo_failures_hide_backend_details(monkeypatch, demo_store, caplog, stage, error_type):
    details = ("/synthetic-private/checkpoint.pth", "http://backend.example.invalid", "DUMMY_CREDENTIAL")

    def fail(*args, **kwargs):
        raise error_type(" ".join(details))

    async def process(*args, **kwargs):
        if stage == "recognition":
            fail()
        return {"metadata": {}, "outputs": {"markdown": "ok"}, "demo_pages": []}

    monkeypatch.setattr(orchestrator, "process_decoded_document", process)
    if stage == "decode":
        monkeypatch.setattr(orchestrator, "_decode_document_images", fail)
    elif stage == "publish":
        monkeypatch.setattr(demo_store, "publish_result", fail)
    with TestClient(app) as client:
        submit = client.post("/v1/demo/jobs", files={"file": ("demo.png", image_bytes(), "image/png")})
        assert submit.status_code == 200
        job_id = submit.json()["data"]["id"]
        state = wait_status(client, job_id, "failed")
        expected = (
            "Temporary storage is unavailable or full. Try again later."
            if error_type is OSError else "Unable to process this document. Please try again later."
        )
        assert state["error"] == expected
        assert client.get(f"/v1/demo/jobs/{job_id}/result").status_code == 409
        persisted = (demo_store.job_dir(job_id) / "manifest.json").read_text()
        assert all(detail not in json.dumps(state) and detail not in persisted for detail in details)
        if error_type is OSError:
            assert state["pages"] == []
            assert not (demo_store.job_dir(job_id) / "images").exists()

    assert "Demo processing failed" in caplog.text
    assert all(detail not in caplog.text for detail in details)


def test_demo_invalid_image_keeps_safe_decode_message(demo_store):
    with TestClient(app) as client:
        submit = client.post("/v1/demo/jobs", files={"file": ("broken.png", b"not-an-image")})
        state = wait_status(client, submit.json()["data"]["id"], "failed")
    assert state["error"] == "Unable to decode document into valid page images."


def test_demo_result_conflict_hides_exception_details(monkeypatch, demo_store):
    def fail(*args, **kwargs):
        raise ValueError("/synthetic-private/result.json DUMMY_CREDENTIAL")

    job_id = demo_store.create("demo.png")["id"]
    monkeypatch.setattr(demo_store, "result", fail)
    with TestClient(app) as client:
        response = client.get(f"/v1/demo/jobs/{job_id}/result")
    assert response.status_code == 409
    assert response.json() == {"code": 409, "message": "The document is not ready yet.", "data": None}


def test_shared_store_heartbeat_preserves_published_manifest(tmp_path):
    owner = DemoStore(tmp_path)
    reader = DemoStore(tmp_path)
    job_id = owner.create("a.png")["id"]
    owner.update(job_id, status="processing", pages=[{"page_num": 0}])
    reader.keep_alive(job_id)
    owner.publish_result(job_id, {"metadata": {}, "outputs": {"markdown": "ok"}})
    reader.keep_alive(job_id)
    assert reader.get(job_id)["status"] == "success"
    assert reader.get(job_id)["pages"] == [{"page_num": 0}]
    assert reader.result(job_id)["outputs"]["markdown"] == "ok"


def test_shared_store_capacity_admission_and_http_unavailable(monkeypatch, demo_store, tmp_path):
    stores = [DemoStore(tmp_path / "limited", max_jobs=1) for _ in range(2)]

    def create(store):
        try:
            return store.create("a.png")
        except OSError:
            return None

    with ThreadPoolExecutor(2) as executor:
        results = list(executor.map(create, stores))
    assert sum(result is not None for result in results) == 1

    monkeypatch.setattr(demo_store, "max_jobs", 0)
    with TestClient(app) as client:
        assert client.post("/v1/demo/jobs", files={"file": ("a.png", image_bytes())}).status_code == 503


def test_shutdown_waits_for_decoder_then_marks_job_failed(monkeypatch, demo_store):
    entered = threading.Event()

    def decode(data, name):
        entered.set()
        time.sleep(0.15)
        return [np.zeros((4, 4, 3), dtype=np.uint8)]

    async def process(*args, **kwargs):
        await asyncio.sleep(10)

    monkeypatch.setattr(orchestrator, "_decode_document_images", decode)
    monkeypatch.setattr(orchestrator, "process_decoded_document", process)
    with TestClient(app) as client:
        response = client.post("/v1/demo/jobs", files={"file": ("a.png", b"a")})
        job_id = response.json()["data"]["id"]
        assert entered.wait(1)
    assert demo_store.get(job_id)["status"] == "failed"
    assert "server stopped" in demo_store.get(job_id)["error"]
