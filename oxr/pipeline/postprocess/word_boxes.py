"""Estimate word boxes from a layout region and its recognized text."""

from __future__ import annotations

import re
import unicodedata

from oxr.pipeline.models import BlockType

# CJK ideographs, kana, hangul syllables, and halfwidth katakana.
_CJK_CHAR = (
    r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af"
    r"\uff66-\uff9d]"
)
_TOKEN_RE = re.compile(
    rf"{_CJK_CHAR}|[^\s\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
    r"\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af\uff66-\uff9d]+"
)

WORD_BLOCK_TYPES = {
    BlockType.TEXT,
    BlockType.TITLE,
    BlockType.CAPTION,
    BlockType.FOOTNOTE,
    BlockType.PAGE_HEADER,
    BlockType.PAGE_FOOTER,
    BlockType.STAMP,
}

_VERTICAL_ASPECT = 1.8


def display_width(text: str) -> float:
    units = 0.0
    for char in text:
        east = unicodedata.east_asian_width(char)
        units += 2.0 if east in {"W", "F"} else 1.0
    return units


def tokenize_line(line: str) -> list[tuple[str, int, int]]:
    """Return (token, start, end) character offsets into the original line."""
    return [(match.group(), match.start(), match.end()) for match in _TOKEN_RE.finditer(line)]


def _round_box(box: list[float]) -> list[float]:
    return [round(value, 2) for value in box]


def _split_lines(content: str) -> list[str]:
    return content.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _layout_horizontal(lines: list[str], x1: float, y1: float, x2: float, y2: float) -> list[dict]:
    line_height = (y2 - y1) / len(lines)
    words: list[dict] = []
    for index, line in enumerate(lines):
        tokens = tokenize_line(line)
        if not tokens:
            continue
        total = display_width(line) or 1.0
        line_y1 = y1 + index * line_height
        line_y2 = y1 + (index + 1) * line_height
        width = x2 - x1
        for text, start, end in tokens:
            word_x1 = x1 + width * (display_width(line[:start]) / total)
            word_x2 = x1 + width * (display_width(line[:end]) / total)
            words.append({"text": text, "bbox": _round_box([word_x1, line_y1, word_x2, line_y2])})
    return words


def _layout_vertical(content: str, x1: float, y1: float, x2: float, y2: float) -> list[dict]:
    tokens = tokenize_line(content)
    if not tokens:
        return []
    total = display_width(content) or 1.0
    height = y2 - y1
    words: list[dict] = []
    for text, start, end in tokens:
        word_y1 = y1 + height * (display_width(content[:start]) / total)
        word_y2 = y1 + height * (display_width(content[:end]) / total)
        words.append({"text": text, "bbox": _round_box([x1, word_y1, x2, word_y2])})
    return words


def estimate_word_boxes(content: str, bbox: list[float]) -> list[dict]:
    """Split recognized text into tokens and allocate boxes inside ``bbox``.

    Latin tokens follow whitespace; CJK glyphs are split per character.
    Multi-line text is stacked top to bottom. A tall, single-line region is
    treated as vertical text. Coordinates stay in the same pixel space as the
    parent region. This is a geometric estimate from the recognized string, not
    a second recognition pass.
    """
    if not content or not str(content).strip() or len(bbox) != 4:
        return []
    x1, y1, x2, y2 = (float(value) for value in bbox)
    if x2 <= x1 or y2 <= y1:
        return []

    lines = _split_lines(str(content))
    vertical = (
        (y2 - y1) > _VERTICAL_ASPECT * (x2 - x1)
        and len(lines) == 1
    )
    if vertical:
        return _layout_vertical(lines[0], x1, y1, x2, y2)
    return _layout_horizontal(lines, x1, y1, x2, y2)
