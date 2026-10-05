"""Verify PDFium exclusion covers open/render/close and releases on errors."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from oxr.utils import pdf


def _png():
    image = np.full((3, 4, 3), 71, dtype=np.uint8)
    success, encoded = cv2.imencode(".png", image)
    assert success
    return image, encoded.tobytes()


@pytest.mark.parametrize("stage", ["open", "render", "close"])
def test_concurrent_documents_exclude_every_pdfium_stage(monkeypatch, stage):
    entered, release, second_started, second_opened = Event(), Event(), Event(), Event()
    expected, png = _png()
    events = []

    def hold(name, current):
        events.append((name, current))
        if name == "first" and current == stage:
            entered.set()
            assert release.wait(2)

    class Document:
        def __init__(self, name):
            self.name = name
            self.pages = [self]

        def __enter__(self):
            return self

        def render(self, *, dpi):
            assert dpi == 150
            hold(self.name, "render")
            return png

        def __exit__(self, *_args):
            hold(self.name, "close")

    def open_document(*, stream):
        name = stream.decode()
        if name == "second":
            second_opened.set()
        hold(name, "open")
        return Document(name)

    opened = Mock(side_effect=open_document)
    monkeypatch.setattr(pdf.sopdf, "open", opened)

    def second_call():
        second_started.set()
        return pdf.pdf_to_images(b"second")

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(pdf.pdf_to_images, b"first")
        try:
            assert entered.wait(1)
            second = pool.submit(second_call)
            assert second_started.wait(1)
            # The first document is deliberately held inside each native stage.
            assert not second_opened.wait(0.05)
            assert not second.done()
            assert opened.call_count == 1
        finally:
            release.set()
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)
    assert events == [(name, current) for name in ["first", "second"]
                      for current in ["open", "render", "close"]]
    assert np.array_equal(first_result[0], expected)
    assert np.array_equal(second_result[0], expected)


@pytest.mark.parametrize("stage", ["open", "render", "close"])
def test_pdfium_lock_is_released_after_native_failure(monkeypatch, stage):
    expected, png = _png()

    class Document:
        def __init__(self, broken):
            self.broken = broken
            self.pages = [self]

        def __enter__(self):
            return self

        def render(self, *, dpi):
            if self.broken and stage == "render":
                raise ValueError("native render failed")
            return png

        def __exit__(self, *_args):
            if self.broken and stage == "close":
                raise ValueError("native close failed")

    def open_document(*, stream):
        if stream == b"broken" and stage == "open":
            raise ValueError("native open failed")
        return Document(stream == b"broken")

    monkeypatch.setattr(pdf.sopdf, "open", open_document)
    with pytest.raises(ValueError, match="native .* failed"):
        pdf.pdf_to_images(b"broken")
    # A non-daemon worker could hang the test process if a broken lock leaked.
    assert pdf._PDFIUM_LOCK.acquire(timeout=1)
    pdf._PDFIUM_LOCK.release()
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(pdf.pdf_to_images, b"healthy").result(timeout=2)
    assert np.array_equal(result[0], expected)
