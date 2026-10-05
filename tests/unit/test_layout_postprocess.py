import math

import pytest

from oxr.pipeline.models import BlockType
from oxr.pipeline.postprocess.layout_postprocess import parse_layout_batch


def payload(labels, boxes, scores, **extra):
    return {
        "status": "success",
        "params": {},
        "data": {"labels": labels, "boxes": boxes, "scores": scores},
        **extra,
    }


def test_parse_layout_batch_maps_all_native_class_ids():
    boxes = [[index, 0, index + 0.5, 1] for index in range(12)]

    pages = parse_layout_batch(
        payload([list(range(12))], [boxes], [[0.5] * 12]),
        [(20, 2)],
        0.5,
    )

    assert [block.type for block in pages[0]] == [
        BlockType.CAPTION,
        BlockType.FOOTNOTE,
        BlockType.FORMULA,
        BlockType.TEXT,
        BlockType.PAGE_FOOTER,
        BlockType.PAGE_HEADER,
        BlockType.PICTURE,
        BlockType.TITLE,
        BlockType.TABLE,
        BlockType.CODE_BLOCK,
        BlockType.STAMP,
        BlockType.CHEMICAL_STRUCTURE,
    ]
    assert [block.idx for block in pages[0]] == list(range(12))
    assert [block.layout_score for block in pages[0]] == [0.5] * 12


def test_parse_layout_batch_applies_classwise_nms_by_default():
    pages = parse_layout_batch(
        payload(
            [[3, 3, 7]],
            [[
                [0, 0, 100, 100],
                [5, 5, 100, 100],
                [5, 5, 100, 100],
            ]],
            [[0.8, 0.9, 0.7]],
        ),
        [(100, 100)],
        0.5,
    )

    assert [block.type for block in pages[0]] == [BlockType.TEXT, BlockType.TITLE]
    assert pages[0][0].bbox == [5.0, 5.0, 100.0, 100.0]
    assert pages[0][0].layout_score == 0.9


def test_parse_layout_batch_can_disable_nms():
    pages = parse_layout_batch(
        payload(
            [[3, 3]],
            [[[0, 0, 100, 100], [5, 5, 100, 100]]],
            [[0.8, 0.9]],
        ),
        [(100, 100)],
        0.5,
        None,
    )

    assert len(pages[0]) == 2


def test_parse_layout_batch_filters_inclusively_and_reindexes_retained_blocks():
    pages = parse_layout_batch(
        payload(
            [[3, 7, 2]],
            [[[-5, 0, 10, 10], [10, 10, 30, 30], [40, 40, 60, 60]]],
            [[0.49, 0.5, 0.9]],
            request_id="ignored",
        ),
        [(100, 100)],
        0.5,
    )

    assert [block.type for block in pages[0]] == [BlockType.TITLE, BlockType.FORMULA]
    assert [block.idx for block in pages[0]] == [0, 1]
    assert [block.bbox for block in pages[0]] == [
        [10.0, 10.0, 30.0, 30.0],
        [40.0, 40.0, 60.0, 60.0],
    ]


def test_parse_layout_batch_preserves_empty_page_positions():
    pages = parse_layout_batch(
        payload([[], [3]], [[], [[1, 2, 30, 40]]], [[], [0.4]]),
        [(100, 100), (100, 100)],
        0.5,
    )

    assert pages == [[], []]


def test_parse_layout_batch_clamps_retained_boxes_to_image_bounds():
    pages = parse_layout_batch(
        payload(
            [[3, 7]],
            [[[-1, -2, 101, 102], [90, 95, 110, 120]]],
            [[0.9, 0.8]],
        ),
        [(100, 100)],
        0.5,
    )

    assert [block.bbox for block in pages[0]] == [
        [0.0, 0.0, 100.0, 100.0],
        [90.0, 95.0, 100.0, 100.0],
    ]


def test_parse_layout_batch_discards_boxes_outside_image_after_clamping():
    pages = parse_layout_batch(
        payload(
            [[3, 7, 2]],
            [[[-20, 1, -10, 10], [101, 1, 110, 10], [1, 101, 10, 110]]],
            [[0.9, 0.9, 0.9]],
        ),
        [(100, 100)],
        0.5,
    )

    assert pages == [[]]


@pytest.mark.parametrize(
    ("response", "image_sizes", "match"),
    [
        (None, [(100, 100)], "response must be an object"),
        ({"status": "failed", "params": {}, "data": {}}, [(100, 100)], "status"),
        ({"status": "success", "params": [], "data": {}}, [(100, 100)], "params"),
        ({"status": "success", "params": {}, "data": []}, [(100, 100)], "data"),
        ({"status": "success", "params": {}, "data": {}}, [(100, 100)], "labels"),
        (payload([[]], [[]], [[]]), [(100, 100), (100, 100)], "batch count"),
        (payload([[3]], [[]], [[0.9]]), [(100, 100)], "candidate count"),
        (payload([[True]], [[[1, 2, 3, 4]]], [[0.9]]), [(100, 100)], "label"),
        (payload([[3.0]], [[[1, 2, 3, 4]]], [[0.9]]), [(100, 100)], "label"),
        (payload([[12]], [[[1, 2, 3, 4]]], [[0.9]]), [(100, 100)], "label"),
        (payload([[3]], [[[1, 2, 3, 4]]], [[True]]), [(100, 100)], "score"),
        (payload([[3]], [[[1, 2, 3, 4]]], [[math.nan]]), [(100, 100)], "finite"),
        (payload([[3]], [[[1, 2, 3, 4]]], [[1.1]]), [(100, 100)], "score"),
        (payload([[3]], [[[1, 2, 3]]], [[0.9]]), [(100, 100)], "four coordinates"),
        (payload([[3]], [[[True, 2, 3, 4]]], [[0.9]]), [(100, 100)], "numeric"),
        (payload([[3]], [[[1, 2, math.inf, 4]]], [[0.9]]), [(100, 100)], "finite"),
        (payload([[3]], [[[3, 2, 3, 4]]], [[0.9]]), [(100, 100)], "nondegenerate"),
        (payload([[3]], [[[4, 2, 3, 4]]], [[0.9]]), [(100, 100)], "nondegenerate"),
    ],
)
def test_parse_layout_batch_rejects_invalid_native_response(response, image_sizes, match):
    with pytest.raises(ValueError, match=match):
        parse_layout_batch(response, image_sizes, 0.5)


@pytest.mark.parametrize("threshold", [True, -0.01, 1.01, math.nan])
def test_parse_layout_batch_rejects_invalid_score_threshold(threshold):
    with pytest.raises(ValueError, match="score threshold"):
        parse_layout_batch(payload([[]], [[]], [[]]), [(100, 100)], threshold)


@pytest.mark.parametrize("threshold", [True, -0.01, 1.01, math.nan])
def test_parse_layout_batch_rejects_invalid_nms_threshold(threshold):
    with pytest.raises(ValueError, match="NMS IoU threshold"):
        parse_layout_batch(payload([[]], [[]], [[]]), [(100, 100)], 0.5, threshold)


@pytest.mark.parametrize("image_size", [(0, 100), (100, 0), (True, 100), (100, math.inf)])
def test_parse_layout_batch_rejects_invalid_image_dimensions(image_size):
    with pytest.raises(ValueError, match="image dimensions"):
        parse_layout_batch(payload([[]], [[]], [[]]), [image_size], 0.5)
