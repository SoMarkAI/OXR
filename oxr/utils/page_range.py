"""Parse and validate 1-based document page selection strings."""

from __future__ import annotations

import re

_RANGE_TOKEN = re.compile(r"^(\d+)(?:-(\d+))?$")
PAGE_RANGE_SYNTAX = (
    "page_range must be a comma-separated list of 1-based pages or inclusive "
    "ranges, such as 1-3,5,8-10."
)


class PageRangeError(ValueError):
    """Invalid page_range syntax or a selection outside the document."""


def parse_page_range(spec: str) -> list[int]:
    """Return sorted unique 0-based page indices from a 1-based range string."""
    text = spec.strip()
    if not text:
        raise PageRangeError("page_range must not be empty.")

    indices: list[int] = []
    for raw_token in text.split(","):
        token = raw_token.strip()
        if not token:
            raise PageRangeError(PAGE_RANGE_SYNTAX)
        match = _RANGE_TOKEN.fullmatch(token)
        if not match:
            raise PageRangeError(PAGE_RANGE_SYNTAX)
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < 1:
            raise PageRangeError("page_range page numbers must be greater than or equal to 1.")
        if end < start:
            raise PageRangeError(f"page_range range {start}-{end} is inverted.")
        indices.extend(range(start - 1, end))
    return sorted(dict.fromkeys(indices))


def resolve_page_indices(page_range: list[int] | None, page_count: int) -> list[int]:
    """Validate 0-based indices against ``page_count``; ``None`` selects every page."""
    if page_count <= 0:
        raise PageRangeError("page_range exceeds document length (0 pages).")
    if page_range is None:
        return list(range(page_count))
    invalid = [index for index in page_range if index < 0 or index >= page_count]
    if invalid:
        raise PageRangeError(f"page_range exceeds document length ({page_count} pages).")
    return list(page_range)
