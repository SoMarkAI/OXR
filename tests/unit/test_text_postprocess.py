import asyncio

import numpy as np

from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.orchestrator import process_document
from oxr.pipeline.postprocess.text_postprocess import postprocess_text


def _text_block() -> Block:
    return Block(0, BlockType.TEXT, [0, 0, 1, 1], "", ContentFormat.TEXT)


def test_text_postprocess_moves_tag_number_outside_inline_formula():
    block = postprocess_text(_text_block(), "$y=x+1 \\tag{1}$")

    assert block.content == "$y=x+1$ (1)"
    assert block.format == ContentFormat.TEXT


def test_text_postprocess_moves_tag_star_number_outside_inline_formula_without_parentheses():
    block = postprocess_text(_text_block(), "$y=x+1 \\tag*{1}$")

    assert block.content == "$y=x+1$ 1"


def test_text_postprocess_moves_leqno_number_before_inline_formula():
    block = postprocess_text(_text_block(), "$y=x+1 \\leqno (1)$")

    assert block.content == "(1) $y=x+1$"


def test_text_postprocess_preserves_math_ranges_without_numbering():
    block = postprocess_text(_text_block(), "Keep $ x + 1 $ unchanged")

    assert block.content == "Keep $ x + 1 $ unchanged"


def test_process_document_cleans_formula_numbering_from_text_blocks(monkeypatch):
    async def fake_analyze_layouts(_images, _model_options=None):
        return [[_text_block()]]

    async def fake_recognize_pages(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        block = pages_blocks[0][0]
        return [[postprocess_text(block, "$y=x+1 \\tag{8.12}$")]]

    monkeypatch.setattr("oxr.pipeline.orchestrator.cv2.imdecode", lambda *_args: np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr("oxr.pipeline.orchestrator.analyze_layouts", fake_analyze_layouts)
    monkeypatch.setattr("oxr.pipeline.orchestrator.recognize_pages", fake_recognize_pages)

    result = asyncio.run(process_document(b"image", "page.png", ["json", "markdown"], keep_header_footer=True))
    blocks = result["outputs"]["json"]["pages"][0]["blocks"]

    assert blocks[0]["type"] == "Text"
    assert blocks[0]["content"] == "$y=x+1$ (8.12)"
    assert result["outputs"]["markdown"] == "$y=x+1$ (8.12)\n"
