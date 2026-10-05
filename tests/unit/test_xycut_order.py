import pytest

from oxr.pipeline._vendor import xycut_plus_plus
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.xycut_order import xycut_plus_order


def _block(index: int, bbox: list[float]) -> Block:
    return Block(index, BlockType.TEXT, bbox, "", ContentFormat.TEXT)


def test_xycut_plus_order_is_order_only_and_returns_exact_permutation():
    blocks = [
        _block(0, [0, 0, 40, 40]),
        _block(1, [50, 0, 90, 40]),
        _block(2, [0, 50, 40, 90]),
        _block(3, [50, 50, 90, 90]),
    ]
    original_bboxes = [block.bbox[:] for block in blocks]

    order = xycut_plus_order(blocks)

    assert order == [0, 2, 1, 3]
    assert sorted(order) == list(range(len(blocks)))
    assert [block.bbox for block in blocks] == original_bboxes


def test_xycut_plus_order_rejects_degenerate_bbox():
    blocks = [
        _block(0, [0, 0, 0, 10]),
        _block(1, [0, 20, 10, 30]),
    ]

    with pytest.raises(ValueError, match="degenerate bbox"):
        xycut_plus_order(blocks)


def test_xycut_plus_order_supports_every_block_type_without_dropping_overlaps():
    typed_blocks = [
        Block(
            index,
            block_type,
            [0, index * 20, 100, index * 20 + 10],
            "",
            ContentFormat.TEXT,
        )
        for index, block_type in enumerate(BlockType)
    ]
    overlapping_blocks = [
        _block(index, [0, 0, 100, 100])
        for index in range(3)
    ]

    typed_order = xycut_plus_order(typed_blocks)
    overlapping_order = xycut_plus_order(overlapping_blocks)

    assert sorted(typed_order) == list(range(len(typed_blocks)))
    assert sorted(overlapping_order) == [0, 1, 2]


def test_xycut_adapter_preserves_vendor_default_overlap_removal():
    entries = [
        {
            "block_bbox": [0, 0, 100, 100],
            "block_label": "text",
            "block_content": str(index),
            "seg_start_flag": False,
            "seg_end_flag": False,
        }
        for index in range(2)
    ]

    ordered = xycut_plus_plus.get_layout_ordering(entries)

    assert len(ordered) == 1
