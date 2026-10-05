import asyncio

import numpy as np
import pytest

from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.orchestrator import process_document
from oxr.pipeline.postprocess.assembly_postprocess import blocks_to_markdown
from oxr.pipeline.postprocess import formula_postprocess
from oxr.pipeline.postprocess.formula_postprocess import (
    clean_formula_markdown_delimiters,
    postprocess_formula,
    space_latex_row_breaks,
    space_markdown_math,
)


@pytest.mark.parametrize(("before", "after"), [
    (r"I= $ $\sum_{n=1}^{N}a_n", r"I=  \sum_{n=1}^{N}a_n"),
    (r"x=$ $1+ $ $2", "x= 1+  2"),
    ("x=$\n$1", "x=\n1"),
    (r"$x=1$", "x=1"),
    (r"x=\text{cost \$5}+$ $y", r"x=\text{cost \$5}+ y"),
    (r"x=\text{$a$}+$ $y", r"x=\text{$a$}+ y"),
])
def test_markdown_delimiter_cleanup_preserves_tokens_lines_and_is_idempotent(before, after):
    assert clean_formula_markdown_delimiters(before) == after
    assert clean_formula_markdown_delimiters(after) == after
    assert before.count("\n") == after.count("\n")


@pytest.mark.parametrize("content", [
    r"x=1$", r"x$y$z", r"x$$ $$y", r"x={$ $y}",
    r"x=1$ and $y=2", r"$x$ 与 $y$", r"\text{price $5}",
    r"x=1\$ \$y=2", r"\verb|$ $|", r"x=\text{unclosed$ $y",
    "% $ $\nx=1", r"x=1$ $y={2", r"a\\[15pt]b",
    r"x=1$ $y=2$ $z=3", "x=1$\n$y=2", r"x=$ $1$ $y=2",
])
def test_markdown_delimiter_cleanup_leaves_ambiguous_or_protected_content_intact(content):
    assert clean_formula_markdown_delimiters(content) == content


def _formula_block() -> Block:
    return Block(0, BlockType.FORMULA, [0, 0, 1, 1], "", ContentFormat.TEXT)


def test_formula_postprocess_strips_inline_dollar_wrappers():
    block = postprocess_formula(_formula_block(), "$x^2 + y^2 = z^2$")

    assert block.content == "x^2 + y^2 = z^2"
    assert block.format == ContentFormat.LATEX


def test_formula_postprocess_strips_display_dollar_wrappers():
    block = postprocess_formula(_formula_block(), "$$\nx^2 + y^2 = z^2\n$$")

    assert block.content == "x^2 + y^2 = z^2"


def test_formula_postprocess_spaces_row_break_before_parenthesis():
    block = postprocess_formula(_formula_block(), r"a&b\\(c)&d")

    assert block.content == r"a&b\\ (c)&d"


def test_markdown_formula_spacing_preserves_code_html_and_incomplete_math():
    markdown = (
        r"$a\\(b)$ `code $c\\(d)$` <span>$e\\(f)$</span> "
        r"$incomplete\\(g)"
    )

    assert space_markdown_math(markdown) == (
        r"$a\\ (b)$ `code $c\\(d)$` <span>$e\\ (f)$</span> "
        r"$incomplete\\(g)"
    )


@pytest.mark.parametrize(
    ("before", "after"),
    [
        (r"a\\(b)", r"a\\ (b)"),
        (r"a\\)b", r"a\\ )b"),
        (r"a\\[2pt](b)", r"a\\[2pt](b)"),
        (r"\(a\)", r"\(a\)"),
        (r"\text{literal {a}\\(b)}\\(x)", r"\text{literal {a}\\(b)}\\ (x)"),
        (r"\text{incomplete\\(b)", r"\text{incomplete\\(b)"),
        (r"\%a\\(b)", r"\%a\\ (b)"),
        ("%a\\\\(b)\nx\\\\(y)", "%a\\\\(b)\nx\\\\ (y)"),
        (r"a\\{b}", r"a\\ {b}"),
        (r"\text{a\\{b}}", r"\text{a\\{b}}"),
    ],
)
def test_latex_row_break_spacing_is_conservative_and_idempotent(before, after):
    assert space_latex_row_breaks(before) == after
    assert space_latex_row_breaks(after) == after


@pytest.mark.parametrize(
    "text",
    [
        r"$$$x\\(y)$$$",
        r"$ x\\(y)$",
        r"$x\\(y) $",
        r"<!-- $x\\(y)$ -->",
        r"``literal ` $a\\(b)$``",
    ],
)
def test_markdown_formula_spacing_leaves_ambiguous_ranges_unchanged(text):
    assert space_markdown_math(text) == text


def test_currency_does_not_pair_across_html_or_hide_later_formula():
    table = "<table><tr><td>10,000 mm btu's,$ per mm btu</td><td>$10</td></tr></table>"
    encoded = "&lt;td&gt;$ per unit&lt;/td&gt;&lt;td&gt;$10&lt;/td&gt;"
    later_formula = r"price $ per unit and later $x\\(y)$"

    assert space_markdown_math(table) == table
    assert space_markdown_math(encoded) == encoded
    assert space_markdown_math(later_formula) == r"price $ per unit and later $x\\ (y)$"

def test_formula_assembly_uses_single_display_math_wrapper():
    block = postprocess_formula(_formula_block(), "$x^2 + y^2 = z^2$")

    markdown = blocks_to_markdown([block])

    assert markdown == "$$\nx^2 + y^2 = z^2\n$$\n"


def test_formula_postprocess_moves_tag_number_after_formula():
    block = postprocess_formula(_formula_block(), "$y=x+1 \\tag{1}$")

    assert block.content == "y=x+1"
    assert getattr(block, "text_after", "") == "(1)"


def test_formula_postprocess_moves_tag_star_number_after_formula_without_parentheses():
    block = postprocess_formula(_formula_block(), "$y=x+1 \\tag*{1}$")

    assert block.content == "y=x+1"
    assert getattr(block, "text_after", "") == "1"


def test_formula_postprocess_moves_eqno_number_after_formula():
    block = postprocess_formula(_formula_block(), "$y=x+1 \\eqno (1)$")

    assert block.content == "y=x+1"
    assert getattr(block, "text_after", "") == "(1)"


def test_formula_postprocess_moves_leqno_number_before_formula():
    block = postprocess_formula(_formula_block(), "$y=x+1 \\leqno (1)$")

    assert block.content == "y=x+1"
    assert getattr(block, "text_before", "") == "(1)"


def test_formula_numbering_expansion_creates_text_blocks_with_contiguous_indexes():
    block = postprocess_formula(_formula_block(), "$y=x+1 \\leqno (L) \\tag{R}$")
    expand = getattr(formula_postprocess, "expand_formula_numbering_blocks", None)

    assert expand is not None
    expanded = expand([block])

    assert [item.idx for item in expanded] == [0, 1, 2]
    assert [item.type for item in expanded] == [
        BlockType.TEXT,
        BlockType.FORMULA,
        BlockType.TEXT,
    ]
    assert [item.content for item in expanded] == ["(L)", "y=x+1", "(R)"]
    assert all(item.bbox == block.bbox for item in expanded)
    assert expanded[0].format == ContentFormat.TEXT
    assert expanded[2].format == ContentFormat.TEXT


def test_formula_numbering_markdown_stays_outside_formula_environment():
    block = postprocess_formula(_formula_block(), "$y=x+1 \\tag{1}$")
    expand = getattr(formula_postprocess, "expand_formula_numbering_blocks", None)

    assert expand is not None
    markdown = blocks_to_markdown(expand([block]))

    assert markdown == "$$\ny=x+1\n$$\n(1)\n"


def test_process_document_keeps_numbering_in_one_json_region_and_unchanged_markdown(monkeypatch):
    async def fake_analyze_layouts(_images, _model_options=None):
        return [[_formula_block()]]

    async def fake_recognize_pages(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        block = pages_blocks[0][0]
        return [[postprocess_formula(block, "$y=x+1 \\tag{1}$")]]

    monkeypatch.setattr("oxr.pipeline.orchestrator.cv2.imdecode", lambda *_args: np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr("oxr.pipeline.orchestrator.analyze_layouts", fake_analyze_layouts)
    monkeypatch.setattr("oxr.pipeline.orchestrator.recognize_pages", fake_recognize_pages)
    monkeypatch.setattr("oxr.config.settings.settings.debug", True)

    result = asyncio.run(process_document(b"image", "page.png", ["json", "markdown"], keep_header_footer=True))
    blocks = result["outputs"]["json"]["pages"][0]["blocks"]

    assert [(block["idx"], block["type"], block["content"], block["format"]) for block in blocks] == [
        (0, "Formula", r"y=x+1\tag{1}", "latex"),
    ]
    assert result["outputs"]["markdown"] == "$$\ny=x+1\n$$\n(1)\n"
    assert result["metadata"]["debug_stats"]["total_blocks"] == 1


@pytest.mark.parametrize("raw", [r"x\tag{48}", r"x\tag*{A}", r"x\leqno (L) \tag{R}"])
def test_numbered_json_round_trip_preserves_numbers_without_mutating_markdown(raw):
    from copy import deepcopy
    from oxr.pipeline.postprocess.json_postprocess import serialize_final_blocks

    block = postprocess_formula(_formula_block(), raw)
    expanded = formula_postprocess.expand_formula_numbering_blocks([block])
    original = deepcopy(expanded)
    expected_md = blocks_to_markdown(expanded)
    records, notes = serialize_final_blocks(expanded)
    assert len(records) == 1 and records[0]["bbox"] == block.bbox and notes == []
    restored = postprocess_formula(_formula_block(), records[0]["content"])
    assert (restored.content, restored.text_before, restored.text_after) == (
        block.content, block.text_before, block.text_after,
    )
    assert expanded == original and blocks_to_markdown(expanded) == expected_md


def test_numbering_without_a_surviving_formula_becomes_a_note_not_a_fake_region():
    from oxr.pipeline.postprocess.json_postprocess import serialize_final_blocks

    expanded = formula_postprocess.expand_formula_numbering_blocks([
        postprocess_formula(_formula_block(), r"x\tag{48}"),
    ])
    records, notes = serialize_final_blocks(expanded[1:])
    assert records == [] and notes == [{"idx": 1, "content": "(48)"}]
