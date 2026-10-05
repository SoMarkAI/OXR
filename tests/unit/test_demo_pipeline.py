import asyncio
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from oxr.pipeline import orchestrator
from oxr.pipeline.assembler import assemble_markdown
from oxr.pipeline.image_assets import ImageAssets, image_assets
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.assembly_postprocess import blocks_to_markdown_with_ranges
from oxr.pipeline.postprocess.formula_postprocess import space_markdown_math
from oxr.pipeline.postprocess.picture_postprocess import postprocess_picture
from oxr.pipeline.postprocess.table_images import image_data_url


def block(idx, kind, content, role=""):
    return Block(idx, kind, [0, 0, 20, 20], content, ContentFormat.MARKDOWN,
                 generated_role=role)


def test_source_ranges_preserve_assembly_and_empty_blocks():
    blocks = [
        block(0, BlockType.TITLE, "Heading"),
        block(2, BlockType.FORMULA, r"a\\(b)"),
        block(3, BlockType.TEXT, "(1)", "formula_numbering"),
        block(5, BlockType.TABLE, "<table>\n<tr><td>A</td></tr>\n</table>"),
        block(6, BlockType.PICTURE, "![](/image.png)"),
        block(9, BlockType.CODE_BLOCK, "```python\nx = 1\n```"),
        block(11, BlockType.FOOTNOTE, "Note"),
        block(12, BlockType.TEXT, " "),
    ]
    markdown, ranges = blocks_to_markdown_with_ranges(list(reversed(blocks)))
    assert space_markdown_math(markdown) == assemble_markdown([blocks])
    lines = markdown.splitlines(keepends=True)
    assert "".join(lines[slice(*ranges[0])]) == "# Heading\n"
    assert "".join(lines[slice(*ranges[2])]) == "$$\na\\\\(b)\n$$\n"
    assert ranges[3][0] == ranges[2][1]  # Tight formula-number separator.
    assert "".join(lines[slice(*ranges[5])]).startswith("<table>")
    assert "".join(lines[slice(*ranges[11])]) == "[^11]: Note\n"
    assert 12 not in ranges


def test_request_scoped_images_are_isolated_and_include_table_images(tmp_path):
    async def save(name):
        token = image_assets.set(ImageAssets(tmp_path / name, f"/assets/{name}/"))
        try:
            image = np.zeros((20, 20, 3), dtype=np.uint8)
            picture = await asyncio.to_thread(
                postprocess_picture, block(0, BlockType.PICTURE, ""), image, "same.png", 0,
            )
            table_url = await asyncio.to_thread(image_data_url, image)
            return picture.content, table_url
        finally:
            image_assets.reset(token)

    async def run():
        return await asyncio.gather(save("one"), save("two"))

    results = asyncio.run(run())
    for name, (picture, table_url) in zip(("one", "two"), results):
        assert f"/assets/{name}/" in picture
        assert table_url.startswith(f"/assets/{name}/")
        assert len(list((tmp_path / name).glob("*.png"))) == 2
    assert image_assets.get() is None


def test_decoded_demo_pipeline_keeps_page_ranges_without_decoding(monkeypatch, tmp_path):
    images = [np.zeros((30, 40, 3), dtype=np.uint8)]

    async def layout(_images, _options):
        return [[block(0, BlockType.TITLE, "Title"), block(1, BlockType.TEXT, "")]]

    async def recognize(blocks, *_args):
        return blocks

    def unexpected_decode(*_args):
        raise AssertionError("Already decoded demo pages must never be decoded again")

    monkeypatch.setattr(orchestrator, "_decode_document_images", unexpected_decode)
    monkeypatch.setattr(orchestrator, "analyze_layouts", layout)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)
    result = asyncio.run(orchestrator.process_decoded_document(
        images, "test.pdf", ["markdown", "json"], demo=True,
        asset_dir=tmp_path, asset_url_prefix="/assets/",
    ))
    page = result["demo_pages"][0]
    assert page["page_size"] == {"w": 40, "h": 30}
    assert page["markdown"] == result["outputs"]["markdown"]
    assert page["blocks"][0]["start_line"] == 0
    assert page["blocks"][0]["end_line"] == 1
    assert page["blocks"][1]["start_line"] is None
    assert image_assets.get() is None


@pytest.mark.parametrize("provides_reading_order", [False, True])
@pytest.mark.parametrize("with_body", [False, True])
@pytest.mark.parametrize("full_overlap_dedupe", [False, True])
def test_demo_header_footer_are_layout_only(
    monkeypatch, provides_reading_order, with_body, full_overlap_dedupe,
):
    header = block(0, BlockType.PAGE_HEADER, "Header must not be rendered")
    header.bbox = [0, 0, 40, 4]
    footer = block(2, BlockType.PAGE_FOOTER, "Footer must not be rendered")
    footer.bbox = [0, 26, 40, 30]
    formula = block(1, BlockType.FORMULA, "x = 1")
    formula.bbox = [0, 8, 20, 18]
    formula.text_before = "(1)"
    formula.text_after = "(2)"
    captured = {}

    async def layout(_images, _options):
        return [[header, formula, footer] if with_body else [header, footer]]

    async def recognize(pages, *_args):
        captured["recognition"] = [[b.type for b in page] for page in pages]
        return pages

    async def reading_order(pages, *_args):
        captured["reading_order"] = [[b.type for b in page] for page in pages]
        return [list(range(len(page))) for page in pages]

    monkeypatch.setattr(orchestrator, "analyze_layouts", layout)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", reading_order)
    monkeypatch.setattr(
        orchestrator.settings.pipeline, "full_overlap_dedupe", full_overlap_dedupe,
    )
    monkeypatch.setattr(
        orchestrator.settings.layout_analyze_model,
        "provides_reading_order", provides_reading_order,
    )
    result = asyncio.run(orchestrator.process_decoded_document(
        [np.zeros((30, 40, 3), dtype=np.uint8)], "test.png", ["markdown", "json"],
        demo=True,
    ))

    expected_inputs = [[BlockType.FORMULA] if with_body else []]
    assert captured["recognition"] == expected_inputs
    if not provides_reading_order:
        assert captured["reading_order"] == expected_inputs
    page = result["demo_pages"][0]
    margins = page["blocks"][-2:]
    assert [b["type"] for b in margins] == ["Page-header", "Page-footer"]
    assert [b["bbox"] for b in margins] == [header.bbox, footer.bbox]
    assert all(b["content"] == "" for b in margins)
    assert all(b["start_line"] is None and b["end_line"] is None for b in margins)
    assert len({b["idx"] for b in page["blocks"]}) == len(page["blocks"])
    assert page["markdown"] == result["outputs"]["markdown"]
    assert "Header" not in page["markdown"] and "Footer" not in page["markdown"]
    body = result["outputs"]["json"]["pages"][0]["blocks"]
    assert all(b["type"] not in {"Page-header", "Page-footer"} for b in body)
    assert [
        {key: value for key, value in b.items() if key not in {"start_line", "end_line"}}
        for b in page["blocks"][:-2]
    ] == body
    if with_body:
        assert [b["idx"] for b in body] == [1]
        assert [b["idx"] for b in margins] == [3, 4]
        assert all(b["start_line"] is not None for b in page["blocks"][:-2])
        assert "(1)" in page["markdown"] and "(2)" in page["markdown"]
        assert body[0]["type"] == "Formula"
        assert body[0]["content"] == r"x = 1\leqno (1)\tag{2}"
        region = page["blocks"][0]
        selected = "\n".join(page["markdown"].splitlines()[
            region["start_line"]:region["end_line"]
        ])
        assert "(1)" in selected and "(2)" in selected and "x = 1" in selected
    else:
        assert page["markdown"] == ""
        assert body == []


def test_demo_orphan_numbering_is_a_note_without_a_layout_region(monkeypatch):
    formula = block(0, BlockType.FORMULA, "x = 1")
    formula.text_after = "(48)"

    async def layout(*_args):
        return [[formula]]

    async def recognize(pages, *_args):
        return pages

    monkeypatch.setattr(orchestrator, "analyze_layouts", layout)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(orchestrator.settings.pipeline, "full_overlap_dedupe", True)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)
    monkeypatch.setattr(orchestrator, "_dedupe_full_page", lambda _raw, blocks, _image: blocks[1:])
    result = asyncio.run(orchestrator.process_decoded_document(
        [np.zeros((30, 40, 3), dtype=np.uint8)], "test.png", ["markdown", "json"],
        demo=True,
    ))
    page = result["demo_pages"][0]
    api_page = result["outputs"]["json"]["pages"][0]
    assert page["blocks"] == api_page["blocks"] == []
    assert page["unattached_formula_numbering"] == api_page["unattached_formula_numbering"] == [
        {"idx": 1, "content": "(48)"},
    ]
    assert page["markdown"] == result["outputs"]["markdown"] == "(48)\n"


def test_markdown_delimiter_repair_preserves_entire_json_and_demo_regions(monkeypatch):
    from oxr.pipeline.postprocess import assembly_postprocess

    formula = block(0, BlockType.FORMULA, r"I=$ $\sum a_n")
    formula.text_after = "(57)"
    text = block(1, BlockType.TEXT, "Keep $5 and code `x$ $y` intact")
    text.bbox = [0, 21, 30, 29]

    async def layout(*_args):
        return [[deepcopy(formula), deepcopy(text)]]

    async def recognize(pages, *_args):
        return pages

    monkeypatch.setattr(orchestrator, "analyze_layouts", layout)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(orchestrator.settings.pipeline, "full_overlap_dedupe", False)
    monkeypatch.setattr(orchestrator.settings.layout_analyze_model, "provides_reading_order", True)

    def run():
        return asyncio.run(orchestrator.process_decoded_document(
            [np.zeros((30, 40, 3), dtype=np.uint8)], "test.png", ["markdown", "json"], demo=True,
        ))

    with patch.object(assembly_postprocess, "clean_formula_markdown_delimiters", side_effect=lambda s: s):
        baseline = run()
    candidate = run()
    assert candidate["outputs"]["json"] == baseline["outputs"]["json"]
    assert candidate["demo_pages"][0]["blocks"] == baseline["demo_pages"][0]["blocks"]
    assert r"I= \sum a_n" in candidate["outputs"]["markdown"]
    assert "(57)" in candidate["outputs"]["markdown"]
    assert text.content in candidate["outputs"]["markdown"]
    assert candidate["demo_pages"][0]["markdown"] == candidate["outputs"]["markdown"]
