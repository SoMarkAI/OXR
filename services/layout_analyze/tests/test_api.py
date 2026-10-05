import numpy as np
from fastapi.testclient import TestClient

from dfine_la.api import create_app


class FakeRuntime:
    def __init__(self, settings):
        self.settings = settings

    def prepare(self, payloads):
        return payloads, 1.0, 2.0

    def infer(self, ready):
        count = len(ready)
        outputs = (
            np.tile(np.array([[0, 1, 1, 2]], dtype=np.int64), (count, 1)),
            np.tile(
                np.array(
                    [[
                        [0.0, 0.0, 1.0, 1.0],
                        [1.0, 1.0, 2.0, 2.0],
                        [2.0, 2.0, 3.0, 3.0],
                        [3.0, 3.0, 4.0, 4.0],
                    ]],
                    dtype=np.float32,
                ),
                (count, 1, 1),
            ),
            np.tile(
                np.array([[0.4, 0.6, 0.8, 0.9]], dtype=np.float32),
                (count, 1),
            ),
        )
        return outputs, {"queue": 0.0, "inference": 3.0}

    def health(self):
        return {"status": "healthy", "cuda_graph_ready": True}

    def info(self):
        return {"endpoint": "/inference"}


def test_api_contract_and_no_versioned_route():
    app = create_app(runtime_factory=FakeRuntime)
    with TestClient(app) as client:
        response = client.post(
            "/inference",
            files=[("images", ("image.png", b"image", "image/png"))],
        )
        assert response.status_code == 200
        assert response.json() == {
            "status": "success",
            "params": {},
            "data": {
                "labels": [[0, 1, 1, 2]],
                "boxes": [[
                    [0.0, 0.0, 1.0, 1.0],
                    [1.0, 1.0, 2.0, 2.0],
                    [2.0, 2.0, 3.0, 3.0],
                    [3.0, 3.0, 4.0, 4.0],
                ]],
                "scores": [[
                    0.4000000059604645,
                    0.6000000238418579,
                    0.800000011920929,
                    0.8999999761581421,
                ]],
            },
        }
        assert "request_read" in response.headers["server-timing"]
        assert client.get("/health").json()["status"] == "healthy"
        assert client.get("/model/info").json()["endpoint"] == "/inference"
        assert client.post("/legacy-inference").status_code == 404
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404


def test_missing_images_returns_422():
    app = create_app(runtime_factory=FakeRuntime)
    with TestClient(app) as client:
        assert client.post("/inference").status_code == 422


def test_global_confidence_filters_all_classes_and_keeps_fields_aligned():
    app = create_app(runtime_factory=FakeRuntime)
    with TestClient(app) as client:
        response = client.post(
            "/inference",
            files=[("images", ("image.png", b"image", "image/png"))],
            data={"confidence": "0.7"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["params"] == {"confidence": 0.7, "class_confidences": {}}
    assert body["data"]["labels"] == [[1, 2]]
    assert body["data"]["boxes"] == [[
        [2.0, 2.0, 3.0, 3.0],
        [3.0, 3.0, 4.0, 4.0],
    ]]
    assert len(body["data"]["scores"][0]) == 2


def test_class_confidences_override_global_confidence():
    app = create_app(runtime_factory=FakeRuntime)
    with TestClient(app) as client:
        response = client.post(
            "/inference",
            files=[("images", ("image.png", b"image", "image/png"))],
            data={
                "confidence": "0.7",
                "class_confidences": '{"1": 0.5, "2": 0.95}',
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["params"] == {
        "confidence": 0.7,
        "class_confidences": {"1": 0.5, "2": 0.95},
    }
    assert body["data"]["labels"] == [[1, 1]]
    assert body["data"]["boxes"] == [[
        [1.0, 1.0, 2.0, 2.0],
        [2.0, 2.0, 3.0, 3.0],
    ]]
    assert len(body["data"]["scores"][0]) == 2


def test_invalid_confidence_parameters_return_422_before_inference():
    app = create_app(runtime_factory=FakeRuntime)
    invalid_data = [
        {"confidence": "-0.1"},
        {"confidence": "nan"},
        {"class_confidences": "[]"},
        {"class_confidences": '{"12": 0.5}'},
        {"class_confidences": '{"1": 1.1}'},
    ]
    with TestClient(app) as client:
        for data in invalid_data:
            response = client.post(
                "/inference",
                files=[("images", ("image.png", b"image", "image/png"))],
                data=data,
            )
            assert response.status_code == 422, data
