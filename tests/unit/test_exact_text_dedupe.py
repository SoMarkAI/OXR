from copy import deepcopy

import pytest

from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.exact_text_dedupe import dedupe_exact_textual_blocks


def _block(kind, score=0.8, **updates):
    fields = dict(
        idx=17, type=kind, bbox=[0.0, 0.0, 100.0, 10.0], content="a",
        format=ContentFormat.TEXT, layout_score=score,
    )
    fields.update(updates)
    return Block(**fields)


def test_triple_keeps_original_winner_and_surrounding_order_without_mutation():
    blocks = [
        _block(BlockType.FORMULA, idx=40),
        _block(BlockType.TEXT, 0.6, idx=30),
        _block(BlockType.CAPTION, 0.7, idx=20),
        _block(BlockType.TITLE, 0.9, idx=10),
        _block(BlockType.TEXT, idx=1, content="other"),
    ]
    before = deepcopy(blocks)
    result = dedupe_exact_textual_blocks(blocks)
    assert [block.idx for block in result] == [40, 10, 1]
    assert all(a is b for a, b in zip(result, [blocks[0], blocks[3], blocks[4]]))
    assert blocks == before
    assert dedupe_exact_textual_blocks(result) is result


@pytest.mark.parametrize("kinds", [
    (BlockType.TEXT, BlockType.CAPTION),
    (BlockType.TEXT, BlockType.TITLE),
    (BlockType.CAPTION, BlockType.TITLE),
])
def test_each_cross_category_pair(kinds):
    blocks = [_block(kinds[0], 0.9), _block(kinds[1], 0.5)]
    assert dedupe_exact_textual_blocks(blocks) == [blocks[0]]


def test_equal_score_uses_list_position_instead_of_idx_or_type_priority():
    blocks = [_block(BlockType.CAPTION, idx=9), _block(BlockType.TITLE, idx=0)]
    assert dedupe_exact_textual_blocks(blocks)[0] is blocks[0]


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), True, "0.9"])
def test_missing_or_invalid_confidence_skips_whole_group(score):
    blocks = [_block(BlockType.TEXT), _block(BlockType.CAPTION, score)]
    assert dedupe_exact_textual_blocks(blocks) is blocks


@pytest.mark.parametrize("field,value", [
    ("bbox", [0, 0, 100, 10.00001]), ("content", "a "), ("content", "A"),
    ("format", ContentFormat.MARKDOWN), ("generated_role", "heading"),
    ("text_before", "# "), ("text_after", "\n"),
])
def test_different_geometry_text_or_assembly_fields_are_preserved(field, value):
    blocks = [_block(BlockType.TEXT), _block(BlockType.CAPTION, **{field: value})]
    assert dedupe_exact_textual_blocks(blocks) is blocks


@pytest.mark.parametrize("bbox", [
    None, [0, 0, 100], [0, 0, 100, float("nan")],
    [0, 0, 100, float("inf")], [0, 0, "100", 10],
    [0, 0, 0, 10], [0, 10, 100, 0],
])
def test_invalid_geometry_is_preserved(bbox):
    blocks = [_block(BlockType.TEXT, bbox=bbox), _block(BlockType.CAPTION, bbox=bbox)]
    assert dedupe_exact_textual_blocks(blocks) is blocks


def test_same_class_multiplicity_does_not_guess_a_winner():
    blocks = [_block(BlockType.TEXT), _block(BlockType.TEXT), _block(BlockType.TITLE)]
    assert dedupe_exact_textual_blocks(blocks) is blocks


def test_unrelated_classes_and_blank_text_are_preserved():
    blocks = [_block(kind) for kind in BlockType if kind != BlockType.CAPTION and kind != BlockType.TITLE]
    assert dedupe_exact_textual_blocks(blocks) is blocks
    blanks = [_block(BlockType.TEXT, content="  "), _block(BlockType.CAPTION, content="  ")]
    assert dedupe_exact_textual_blocks(blanks) is blanks


def test_invalid_group_does_not_block_an_independent_valid_pair():
    blocks = [
        _block(BlockType.TEXT, None), _block(BlockType.CAPTION),
        _block(BlockType.TITLE, 0.9, content="second"),
        _block(BlockType.TEXT, 0.2, content="second"),
    ]
    assert dedupe_exact_textual_blocks(blocks) == blocks[:3]
