from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.cross_category_dedupe import (
    dedupe_systematic_cross_category_blocks,
    normalize_cross_category_content,
)


def _block(
    idx: int,
    block_type: BlockType,
    bbox: list[float],
    content: str,
    score: float,
) -> Block:
    return Block(
        idx=idx,
        type=block_type,
        bbox=bbox,
        content=content,
        format=ContentFormat.LATEX if block_type == BlockType.FORMULA else ContentFormat.TEXT,
        layout_score=score,
    )


def _systematic_page(extra_text_blocks: int = 6) -> list[Block]:
    blocks = [
        _block(0, BlockType.TEXT, [0, 0, 10, 10], "$x$", 0.9),
        _block(1, BlockType.FORMULA, [0, 0, 10, 10], "x", 0.2),
        _block(2, BlockType.TEXT, [0, 20, 10, 30], r"\[y\]", 0.3),
        _block(3, BlockType.FORMULA, [0, 20, 10, 30], "y", 0.8),
    ]
    blocks.extend(
        _block(
            4 + index,
            BlockType.TEXT,
            [20, index, 30, index + 1],
            f"unique {index}",
            0.9,
        )
        for index in range(extra_text_blocks)
    )
    return blocks


def test_normalizes_one_outer_math_wrapper_and_whitespace():
    assert normalize_cross_category_content(" $$\n x + y \n$$ ") == "x+y"
    assert normalize_cross_category_content(r"\[ x + y \]") == "x+y"
    assert normalize_cross_category_content("$$$x$$$") == "$x$"


def test_dedupes_multiple_dense_groups_by_score():
    result = dedupe_systematic_cross_category_blocks(_systematic_page())

    assert [block.idx for block in result] == [0, 3, 4, 5, 6, 7, 8, 9]


def test_equal_score_preserves_earlier_reading_order_position():
    blocks = _systematic_page()
    blocks[0].layout_score = blocks[1].layout_score = 0.9

    result = dedupe_systematic_cross_category_blocks(blocks)

    assert blocks[0] in result
    assert blocks[1] not in result


def test_isolated_or_sparse_duplicates_are_unchanged():
    isolated = _systematic_page()[:2]
    sparse = _systematic_page(extra_text_blocks=17)

    assert dedupe_systematic_cross_category_blocks(isolated) is isolated
    assert dedupe_systematic_cross_category_blocks(sparse) is sparse


def test_requires_cross_category_exact_bbox_and_content():
    blocks = _systematic_page()
    blocks[1].bbox = [0, 0, 11, 10]
    blocks[3].content = "z"

    assert dedupe_systematic_cross_category_blocks(blocks) is blocks


def test_missing_layout_scores_tie_at_zero():
    blocks = _systematic_page()
    for block in blocks[:4]:
        block.layout_score = None

    result = dedupe_systematic_cross_category_blocks(blocks)

    assert [block.idx for block in result[:2]] == [0, 2]
