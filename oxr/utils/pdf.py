import threading
from collections.abc import Sequence

import sopdf
import numpy as np
import cv2

from oxr.utils.page_range import PageRangeError, resolve_page_indices


# PDFium requires process-wide exclusion, even for separate documents.
_PDFIUM_LOCK = threading.Lock()


def _decode_png(png_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(png_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def pdf_to_images(
    pdf_bytes: bytes,
    dpi: int = 150,
    *,
    page_indices: Sequence[int] | None = None,
) -> list[np.ndarray]:
    """Convert PDF bytes to a list of OpenCV images (numpy arrays, BGR format)."""
    images, _total = render_pdf_pages(pdf_bytes, dpi=dpi, page_indices=page_indices)
    return images


def render_pdf_pages(
    pdf_bytes: bytes,
    dpi: int = 150,
    *,
    page_indices: Sequence[int] | None = None,
) -> tuple[list[np.ndarray], int]:
    """Render selected PDF pages and return ``(images, total_page_count)``.

    ``page_indices`` are 0-based. When omitted, every page is rendered in order.
    """
    with _PDFIUM_LOCK:
        images = []
        with sopdf.open(stream=pdf_bytes) as doc:
            pages = list(doc.pages)
            total = len(pages)
            try:
                selected = resolve_page_indices(
                    None if page_indices is None else list(page_indices),
                    total,
                )
            except PageRangeError:
                raise
            for index in selected:
                png_bytes = pages[index].render(dpi=dpi)
                images.append(_decode_png(png_bytes))
        return images, total
