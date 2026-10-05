from copy import deepcopy

import pytest
import numpy as np

from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.full_overlap_dedupe import dedupe_full_overlaps


def _block(kind=BlockType.TEXT, box=None, score=0.8, idx=17, content="different"):
    return Block(
        idx, kind, [0, 0, 10, 10] if box is None else box, content,
        ContentFormat.TEXT, layout_score=score,
    )


@pytest.mark.parametrize("kind", list(BlockType))
def test_containment_requires_a_parent_with_verified_textual_content(kind):
    outer = _block(kind, [0, 0, 100, 100], 0.2)
    inner = _block(BlockType.FORMULA, [20, 20, 60, 60], 0.99)
    expected = [inner, outer] if kind in {BlockType.PICTURE, BlockType.TABLE} else [outer]
    assert dedupe_full_overlaps([inner, outer]) == expected


@pytest.mark.parametrize("kind", list(BlockType))
def test_equal_geometry_across_every_category_uses_confidence_for_duplicates(kind):
    lower = _block(kind, score=0.2, content="same text")
    higher = _block(BlockType.TEXT, score=0.9, content="same text")
    expected = [lower, higher] if kind in {BlockType.CODE_BLOCK, BlockType.TABLE} else [higher]
    assert dedupe_full_overlaps([lower, higher]) == expected


@pytest.mark.parametrize("kind", list(BlockType))
def test_tiny_overlap_across_every_category_keeps_both_blocks(kind):
    a = _block(kind, [0, 0, 10, 10], 0.2)
    b = _block(BlockType.FORMULA, [9, 9, 20, 20], 0.9)
    assert dedupe_full_overlaps([a, b]) == [a, b]


def test_formula_text_overlap_uses_confidence_despite_distinct_recognized_content():
    formula = _block(BlockType.FORMULA, score=0.7, content="x^{2}")
    text = _block(BlockType.TEXT, score=0.9, content="incorrect but higher confidence")
    assert dedupe_full_overlaps([formula, text]) == [text]
    formula.layout_score = 0.99
    assert dedupe_full_overlaps([formula, text]) == [formula]


@pytest.mark.parametrize("outer_kind, inner_kind", [
    (BlockType.TEXT, BlockType.FORMULA),
    (BlockType.FORMULA, BlockType.TEXT),
    (BlockType.FORMULA, BlockType.FORMULA),
])
def test_formula_containment_does_not_require_recognized_content_inclusion(outer_kind, inner_kind):
    outer = _block(outer_kind, [0, 0, 100, 100], 0.2, content="parent OCR")
    inner = _block(inner_kind, [10, 10, 90, 30], 0.99, content="different child OCR")
    assert dedupe_full_overlaps([inner, outer]) == [outer]


def test_formula_overlap_exemption_keeps_code_and_generated_numbers():
    formula = _block(BlockType.FORMULA, score=0.99, content="x^{2}")
    code = _block(BlockType.CODE_BLOCK, score=0.7, content="unique code")
    number = _block(BlockType.TEXT, score=0.7, content="(48)")
    number.generated_role = "formula_numbering"
    assert dedupe_full_overlaps([code, formula, number]) == [code, formula, number]
    code.layout_score = 1.0
    assert dedupe_full_overlaps([code, formula, number]) == [code, formula, number]


def test_nested_containment_retains_text_not_verified_by_visual_parent():
    blocks = [
        _block(BlockType.FORMULA, [10, 10, 20, 20], 0.99),
        _block(BlockType.PICTURE, [0, 0, 100, 100], None),
        _block(BlockType.CAPTION, [5, 5, 30, 30], 0.9),
    ]
    assert dedupe_full_overlaps(blocks) == [blocks[1], blocks[2]]


def test_outer_box_wins_when_child_shares_some_boundaries():
    outer = _block(box=[0, 0, 10, 20], score=0.1)
    inner = _block(box=[0, 0, 10, 10], score=1.0)
    assert dedupe_full_overlaps([inner, outer]) == [outer]


def test_containment_does_not_make_tiny_overlaps_delete_parents():
    outer = _block(box=[0, 0, 10, 10], score=0.2)
    inner = _block(box=[0, 0, 2, 2], score=0.99)
    crossing = _block(box=[9, 9, 20, 20], score=0.9)
    assert dedupe_full_overlaps([outer, inner, crossing]) == [outer, crossing]


def test_suppressed_intermediary_cannot_suppress_nonoverlapping_neighbor():
    # A-B and B-C qualify as duplicates; A-C does not. Only survivors suppress.
    a = _block(box=[0, 0, 100, 100], score=0.9)
    b = _block(box=[5, 0, 105, 100], score=0.8)
    c = _block(box=[10, 0, 110, 100], score=0.7)
    assert dedupe_full_overlaps([c, b, a]) == [c, a]


def test_ties_keep_page_position_instead_of_smaller_idx():
    a = _block(BlockType.TEXT, idx=99)
    b = _block(BlockType.FORMULA, idx=0)
    assert dedupe_full_overlaps([a, b]) == [a]


@pytest.mark.parametrize("box", [[10, 0, 20, 10], [10, 10, 20, 20], [20, 0, 30, 10]])
def test_touching_or_disjoint_boxes_are_preserved(box):
    blocks = [_block(), _block(box=box)]
    assert dedupe_full_overlaps(blocks) is blocks


@pytest.mark.parametrize("score", [None, float("nan"), float("inf"), True, "0.9"])
@pytest.mark.parametrize("box", [[0, 0, 10, 10], [9, 0, 19, 10]])
def test_unknown_or_invalid_scores_are_not_guessed(score, box):
    blocks = [_block(score=1.0), _block(box=box, score=score)]
    assert dedupe_full_overlaps(blocks) is blocks


@pytest.mark.parametrize("box", [
    [0, 0, 0, 10], [10, 0, 0, 10], [0, 0, 10],
    [0, 0, "10", 10], [0, 0, True, 10], [0, 0, 10, float("nan")],
    [0, 0, 10, float("inf")],
])
def test_invalid_geometry_cannot_delete_valid_blocks(box):
    blocks = [_block(box=box, score=1.0), _block(score=0.1)]
    assert dedupe_full_overlaps(blocks) is blocks


def test_retained_fields_references_and_page_order_are_unchanged():
    blocks = [
        _block(box=[20, 0, 30, 10], idx=88),
        _block(BlockType.FORMULA, score=1.0, idx=5, content="x"),
        _block(BlockType.TEXT, score=0.2, idx=6, content="x"),
    ]
    blocks[1].format = ContentFormat.LATEX
    blocks[1].text_after = "(1)"
    before = deepcopy(blocks)
    result = dedupe_full_overlaps(blocks)
    assert result[0] is blocks[0] and result[1] is blocks[1]
    assert blocks == before
    assert dedupe_full_overlaps(result) is result


@pytest.mark.parametrize("box, suppressed", [
    ([5, 0, 105, 100], True),
    ([5.01, 0, 105.01, 100], False),
    ([0.01, 0, 124.9875, 100], True),
    ([0.01, 0, 124.99, 100], False),
])
def test_overlap_requires_both_iou_and_smaller_coverage_thresholds(box, suppressed):
    a = _block(BlockType.PICTURE, box=[0, 0, 100, 100], score=0.2)
    b = _block(BlockType.PICTURE, box=box, score=0.9)
    image = np.full((100, 130), 255, dtype=np.uint8)
    image[40:60, 40:60] = 0
    result = dedupe_full_overlaps([a, b], image)
    assert result == ([b] if suppressed else [a, b])


def test_children_reappear_when_their_parent_is_suppressed():
    parent = _block(BlockType.PICTURE, [0, 0, 100, 100], 0.2)
    child = _block(BlockType.PICTURE, [99.1, 10, 99.9, 20], 0.99)
    crossing = _block(BlockType.PICTURE, [-1, 0, 99, 100], 0.9)
    image = np.full((100, 100), 255, dtype=np.uint8)
    image[40:60, 40:60] = 0
    assert dedupe_full_overlaps([parent, child, crossing], image) == [child, crossing]


@pytest.mark.parametrize("kind", [BlockType.TEXT, BlockType.CAPTION, BlockType.TITLE, BlockType.CHEMICAL_STRUCTURE, BlockType.FORMULA])
@pytest.mark.parametrize("visual", [BlockType.PICTURE, BlockType.TABLE])
def test_textual_boxes_cannot_preserve_independent_visual_content(kind, visual):
    parent = _block(kind, [0, 0, 100, 100], 0.95, content="Section heading")
    child = _block(visual, [10, 10, 90, 90], 0.8, content="Unique visual content")
    assert dedupe_full_overlaps([parent, child]) == [parent, child]
    child.bbox = parent.bbox.copy()
    expected = [parent] if visual == BlockType.PICTURE and kind != BlockType.CHEMICAL_STRUCTURE else [parent, child]
    assert dedupe_full_overlaps([parent, child]) == expected


def test_tables_require_identical_complete_content_and_structure():
    a = _block(BlockType.TABLE, score=0.9, content='<table><tr><td>123</td></tr></table>')
    b = _block(BlockType.TABLE, score=0.8, content=a.content)
    assert dedupe_full_overlaps([a, b]) == [a]
    b.content = '<table><tr><td colspan="2">123</td></tr></table>'
    assert dedupe_full_overlaps([a, b]) == [a, b]
    b.content = a.content
    b.text_after = 'Unique note'
    assert dedupe_full_overlaps([a, b]) == [a, b]


def test_shifted_picture_requires_ink_coverage_not_just_geometry():
    left = _block(BlockType.PICTURE, [0, 0, 100, 100], 0.8, content='![](left.png)')
    right = _block(BlockType.PICTURE, [5, 0, 105, 100], 0.9, content='![](right.png)')
    image = np.full((100, 105, 3), 255, dtype=np.uint8)
    image[40:60, 50:60] = 0
    assert dedupe_full_overlaps([left, right]) == [left, right]
    assert dedupe_full_overlaps([left, right], image) == [right]
    image[40:60, 0:4] = 0
    assert dedupe_full_overlaps([left, right], image) == [left, right]


def test_complete_text_containment_ignores_recognition_differences():
    outer = _block(box=[0, 0, 100, 100], content="题干")
    child = _block(box=[10, 10, 50, 20], content="A B C D")
    empty = _block(box=[10, 30, 50, 40], content="")
    assert dedupe_full_overlaps([outer, child, empty]) == [outer]


@pytest.mark.parametrize("scale", [0.25, 1.0, 4.0])
def test_height_relative_tolerance_is_scale_invariant_and_content_proven(scale):
    outer = _block(box=[0, 0, 100, 100], content="题干 选项")
    child = _block(box=[-2, 10, 50, 20], content="选 项")
    for block in [outer, child]:
        block.bbox = [coordinate * scale for coordinate in block.bbox]
    assert dedupe_full_overlaps([outer, child]) == [outer]
    child.bbox[0] = -2.01 * scale
    assert dedupe_full_overlaps([outer, child]) == [outer, child]
    child.bbox[0] = -1 * scale
    child.content = "另一行"
    assert dedupe_full_overlaps([outer, child]) == [outer, child]


def test_near_containment_requires_coverage_and_preserves_the_formula_exemption():
    parent = _block(box=[0, 0, 100, 100], content="x")
    child = _block(BlockType.FORMULA, [-2, 10, 4, 30], content="x")
    assert dedupe_full_overlaps([parent, child]) == [parent, child]
    child.bbox = [-2, 10, 8, 30]
    assert dedupe_full_overlaps([parent, child]) == [parent]
    child.content = "y"
    assert dedupe_full_overlaps([parent, child]) == [parent]
    child.type = BlockType.TEXT
    assert dedupe_full_overlaps([parent, child]) == [parent, child]


def test_complete_containment_does_not_bypass_code_or_number_protection():
    parent = _block(box=[0, 0, 100, 100], content="正文")
    code = _block(BlockType.CODE_BLOCK, [10, 10, 50, 30], content="unique code")
    number = _block(BlockType.TEXT, [10, 40, 50, 50], content="(48)")
    number.generated_role = "formula_numbering"
    assert dedupe_full_overlaps([parent, code, number]) == [parent, code, number]


def test_equivalent_chemical_equation_notations_are_deduplicated():
    canonical = _block(score=0.9, content=r"\ce{C6H12O6 -> 2C2H5OH + 2CO2}")
    text = _block(score=0.7, content=r"$\mathrm{C}_{6}\mathrm{H}_{12}\mathrm{O}_{6}\to2\mathrm{C}_{2}\mathrm{H}_{5}\mathrm{OH}+2\mathrm{CO}_{2}$")
    assert dedupe_full_overlaps([canonical, text]) == [canonical]
    text.content = text.content.replace("2\\mathrm{CO}", "3\\mathrm{CO}")
    assert dedupe_full_overlaps([canonical, text]) == [canonical, text]


def test_image_paths_cannot_prove_text_coverage_and_code_indentation_is_preserved():
    picture = _block(BlockType.PICTURE, [0, 0, 100, 100], content='![](/tmp/A.png)')
    text = _block(box=[10, 10, 20, 20], content='A')
    assert dedupe_full_overlaps([picture, text]) == [picture, text]
    a = _block(BlockType.CODE_BLOCK, score=0.9, content='if ok:\n    run()\nfinish()')
    b = _block(BlockType.CODE_BLOCK, score=0.7, content='if ok:\n    run()\n    finish()')
    assert dedupe_full_overlaps([a, b]) == [a, b]


@pytest.mark.parametrize("kind", [BlockType.CAPTION, BlockType.TEXT, BlockType.FORMULA])
def test_chemical_structure_cannot_absorb_independent_contained_text(kind):
    structure = _block(BlockType.CHEMICAL_STRUCTURE, [0, 0, 100, 100], content=r"$\smiles{CC(=O)O}$")
    label = _block(kind, [10, 70, 40, 90], content="(a) Aspirin")
    assert dedupe_full_overlaps([structure, label]) == [structure, label]


@pytest.mark.parametrize("kind", [BlockType.TEXT, BlockType.TITLE, BlockType.CAPTION, BlockType.FORMULA, BlockType.CODE_BLOCK])
@pytest.mark.parametrize("scores", [(0.99, 0.1), (None, None)])
def test_exact_picture_text_alias_prefers_text_independent_of_confidence(kind, scores):
    picture = _block(BlockType.PICTURE, score=scores[0], content="![](label.png)")
    text = _block(kind, score=scores[1], content="例3 变式4")
    assert dedupe_full_overlaps([picture, text]) == [text]
    assert dedupe_full_overlaps([text, picture]) == [text]
    text.bbox[2] += 0.01
    assert dedupe_full_overlaps([picture, text]) == [picture, text]


def test_picture_alias_requires_surviving_nonempty_text_not_generated_number():
    picture = _block(BlockType.PICTURE, score=0.99, content="![](label.png)")
    text = _block(score=0.7, content=" ")
    assert dedupe_full_overlaps([picture, text]) == [picture, text]
    text.content = "(48)"
    text.generated_role = "formula_numbering"
    assert dedupe_full_overlaps([picture, text]) == [picture, text]
    text.generated_role = ""
    text.content = "same text"
    crossing = _block(box=[0.1, 0, 10.1, 10], score=0.9, content="same text")
    assert dedupe_full_overlaps([picture, text, crossing]) == [picture, crossing]
