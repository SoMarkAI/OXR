import asyncio

import pytest

from oxr.config.settings import settings
from oxr.server.app import app, lifespan


@pytest.mark.parametrize(
    ("layout_url", "oxr_url", "missing_name"),
    [
        (None, "http://oxr:8080/v1", "layout_analyze_model.url"),
        ("http://layout:8081", None, "oxr_model.url"),
    ],
)
def test_direct_asgi_lifespan_requires_both_model_urls(
    monkeypatch,
    layout_url,
    oxr_url,
    missing_name,
):
    monkeypatch.setattr(settings.layout_analyze_model, "url", layout_url)
    monkeypatch.setattr(settings.oxr_model, "url", oxr_url)

    async def scenario():
        with pytest.raises(ValueError, match=missing_name):
            async with lifespan(app):
                pass

    asyncio.run(scenario())


def test_lifespan_closes_both_clients_when_one_close_fails(monkeypatch):
    closed = []
    monkeypatch.setattr(settings.layout_analyze_model, "url", "http://layout:8081")
    monkeypatch.setattr(settings.oxr_model, "url", "http://oxr:8080/v1")

    async def close_layout():
        closed.append("layout")
        raise RuntimeError("layout close failed")

    async def close_oxr():
        closed.append("oxr")

    monkeypatch.setattr("oxr.server.app.layout_client.aclose", close_layout)
    monkeypatch.setattr("oxr.server.app.model_client.aclose", close_oxr)

    async def scenario():
        with pytest.raises(RuntimeError, match="layout close failed"):
            async with lifespan(app):
                pass

    asyncio.run(scenario())

    assert closed == ["layout", "oxr"]
