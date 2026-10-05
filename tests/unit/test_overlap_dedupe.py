from dataclasses import asdict

import numpy as np
import pytest

from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.overlap_dedupe import dedupe_complete_text, dedupe_visual_parents


def block(idx, kind, box, text='', score=0.9):
    return Block(idx, kind, box, text, ContentFormat.TEXT, layout_score=score)


def text_group(separator='\n'):
    return [
        block(42, BlockType.TEXT, [0, 0, 100, 100], ' hello' + separator + 'world '),
        block(91, BlockType.TEXT, [0, 0, 100, 40], 'hello'),
        block(3, BlockType.TEXT, [0, 50, 100, 90], 'world'),
    ]


@pytest.mark.parametrize('separator,expected', [('\n', ' hello\n\nworld '), (' ', ' hello world '), ('', ' helloworld ')])
def test_text_preserves_parent_boundaries_native_ids_and_roles(separator, expected):
    blocks = text_group(separator)
    blocks[0].generated_role = 'formula_number'
    before = [asdict(b) for b in blocks]
    result = dedupe_complete_text(blocks)
    assert len(result) == 1
    assert asdict(result[0]) == {**before[0], 'content': expected}
    assert [asdict(b) for b in blocks] == before


@pytest.mark.parametrize('case', ['unique_text', 'interleaved', 'reversed', 'title', 'short'])
def test_text_skips_incomplete_or_unsafe_groups(case):
    blocks = text_group()
    if case == 'unique_text':
        blocks[0].content += ' missing'
    elif case == 'interleaved':
        blocks.insert(2, block(18, BlockType.FORMULA, [0, 0, 1, 1], 'x'))
    elif case == 'reversed':
        blocks[1], blocks[2] = blocks[2], blocks[1]
    elif case == 'title':
        blocks[2].type = BlockType.TITLE
    else:
        blocks[1].content = 'hi'
    assert dedupe_complete_text(blocks) == blocks


def visual_group(kind):
    parent = block(35, kind, [0, 0, 100, 100], 'parent', 0.6)
    children = [block(8, kind, [0, 0, 45, 90], 'one'), block(17, kind, [55, 0, 100, 90], 'two')]
    caption = block(29, BlockType.CAPTION, [0, 90, 100, 100], 'caption')
    text = block(28, BlockType.TEXT, [2, 2, 10, 10], 'inside image')
    image = np.full((100, 100, 3), 255, np.uint8)
    image[10:20, 10:20] = 0
    image[10:20, 65:75] = 0
    image[95:98, 10:80] = 0
    return [parent, *children, caption, text], image


@pytest.mark.parametrize('kind', [BlockType.PICTURE, BlockType.CHEMICAL_STRUCTURE])
def test_visual_parent_removed_with_captions_and_all_retained_fields_intact(kind):
    raw, image = visual_group(kind)
    final = [raw[2], raw[0], raw[4], raw[1], raw[3]]
    before = [asdict(b) for b in final]
    result = dedupe_visual_parents(raw, final, image)
    assert [asdict(b) for b in result] == [b for b in before if b['idx'] != 35]
    assert [asdict(b) for b in final] == before


@pytest.mark.parametrize('case', ['uncovered_ink', 'low_confidence', 'missing_score', 'missing_child', 'missing_caption', 'duplicate_final', 'duplicate_raw', 'blank', 'one_child', 'formula'])
def test_visual_parent_requires_complete_unambiguous_evidence(case):
    raw, image = visual_group(BlockType.PICTURE)
    final = list(raw)
    if case == 'uncovered_ink':
        image[10:30, 48:52] = 0
    elif case == 'low_confidence':
        raw[1].layout_score = 0.61
    elif case == 'missing_score':
        raw[0].layout_score = None
    elif case == 'missing_child':
        final.remove(raw[1])
    elif case == 'missing_caption':
        final.remove(raw[3])
    elif case == 'duplicate_final':
        final.append(raw[1])
    elif case == 'duplicate_raw':
        raw[1].idx = raw[2].idx
    elif case == 'blank':
        image[:] = 255
    elif case == 'one_child':
        raw.remove(raw[2])
    else:
        for b in raw[:3]:
            b.type = BlockType.FORMULA
    assert dedupe_visual_parents(raw, final, image) == final


def test_postprocessing_is_enabled_by_code_local_constants():
    from oxr.pipeline import orchestrator

    assert orchestrator._SYSTEMATIC_CROSS_CATEGORY_DEDUPE is True
    assert orchestrator._VISUAL_PARENT_DEDUPE is True
    assert orchestrator._COMPLETE_TEXT_DEDUPE is True


@pytest.mark.parametrize('enabled', [True, False])
def test_pipeline_cleans_final_blocks_after_original_recognition_and_order(monkeypatch, enabled):
    import asyncio
    from oxr.config.settings import settings
    from oxr.pipeline import orchestrator

    raw_visual, image = visual_group(BlockType.PICTURE)
    raw = raw_visual + text_group()
    # Give the second group distinct native IDs without changing its list order.
    for i, b in enumerate(raw[5:], 100):
        b.idx = i
    original_ids = [b.idx for b in raw]
    events = []

    async def layout(_images, _options):
        return [raw]

    async def recognize(pages, *_args):
        assert [b.idx for b in pages[0]] == original_ids
        events.append('recognize')
        # Recognition may mutate its input: visual evidence must be snapshotted.
        pages[0][0].layout_score = None
        return pages

    async def order(pages, *_args):
        assert [b.idx for b in pages[0]] == original_ids
        events.append('order')
        return [list(range(len(raw)))]

    expand = orchestrator.expand_formula_numbering_blocks

    def expansion(blocks):
        events.append('expand')
        return expand(blocks)

    monkeypatch.setattr(orchestrator, '_decode_document_images', lambda *_: [image])
    monkeypatch.setattr(orchestrator, 'analyze_layouts', layout)
    monkeypatch.setattr(orchestrator, 'recognize_pages', recognize)
    monkeypatch.setattr(orchestrator, 'resolve_reading_orders', order)
    monkeypatch.setattr(orchestrator, 'expand_formula_numbering_blocks', expansion)
    monkeypatch.setattr(settings.layout_analyze_model, 'provides_reading_order', False)
    monkeypatch.setattr(orchestrator, '_SYSTEMATIC_CROSS_CATEGORY_DEDUPE', False)
    monkeypatch.setattr(orchestrator, '_VISUAL_PARENT_DEDUPE', enabled)
    monkeypatch.setattr(orchestrator, '_COMPLETE_TEXT_DEDUPE', enabled)
    result = asyncio.run(orchestrator.process_document(b'fixture', 'fixture.png', ['json']))
    final = result['outputs']['json']['pages'][0]['blocks']
    assert events[-1] == 'expand'
    # Existing formula expansion renumbers first; cleanup preserves those IDs.
    assert {b['idx'] for b in final} == set(range(len(raw))) - ({0, 6, 7} if enabled else set())
    if enabled:
        assert next(b for b in final if b['idx'] == 5)['content'] == ' hello\n\nworld '


@pytest.mark.parametrize('parent,child', [
    ('New York is lovely', 'NewYork is lovely'),
    (r'Visit $\text{New York}$ now', r'Visit $\text{NewYork}$ now'),
    ('NewYork is lovely', 'New York is lovely'),
    ('Keep\tthese  spaces', 'Keep these spaces'),
])
def test_parent_internal_whitespace_is_authoritative(parent, child):
    blocks = [block(0, BlockType.TEXT, [0, 0, 100, 100], parent),
              block(1, BlockType.TEXT, [0, 0, 100, 40], child)]
    result = dedupe_complete_text(blocks)
    assert len(result) == 1
    assert result[0].content == parent


@pytest.mark.parametrize('first,second', [
    (r'Visit $\text{New', 'York}$ now'),
    ('Use `New', 'York` now'),
    ('Visit [New', 'York](https://example.test)'),
    ('<span>New', 'York</span>'),
    ('**New', 'York**'),
    ('Heading', '====='),
    ('Heading', '-----'),
    ('hello  ', 'world'),
    ('- New first', '- New second'),
    ('    code first', '    code second'),
    (r'Visit \(New', r'York\) now'),
])
def test_structured_text_keeps_original_boundary_and_parent_content(first, second):
    parent = first + '\n' + second
    blocks = [block(0, BlockType.TEXT, [0, 0, 100, 100], parent),
              block(1, BlockType.TEXT, [0, 0, 100, 40], first),
              block(2, BlockType.TEXT, [0, 50, 100, 90], second)]
    result = dedupe_complete_text(blocks)
    assert len(result) == 1
    assert result[0].content == parent


def test_multichild_preserves_internal_spaces_and_plain_paragraph_boundary():
    blocks = [block(0, BlockType.TEXT, [0, 0, 100, 100], '  New York\nLos Angeles  '),
              block(1, BlockType.TEXT, [0, 0, 100, 40], 'NewYork'),
              block(2, BlockType.TEXT, [0, 50, 100, 90], 'LosAngeles')]
    result = dedupe_complete_text(blocks)
    assert result[0].content == '  New York\n\nLos Angeles  '
    assert len(result) == 1


@pytest.mark.parametrize('case', ['complete', 'uncovered_ink', 'missing_child', 'missing_title'])
def test_mixed_visual_parent_requires_retained_children_and_complete_ink(case):
    raw, image = visual_group(BlockType.PICTURE)
    raw[1].type = raw[2].type = BlockType.CHEMICAL_STRUCTURE
    raw[3].type = BlockType.TITLE
    lower = block(37, BlockType.CHEMICAL_STRUCTURE, [45, 0, 55, 30], 'three', 0.5)
    raw.append(lower)
    image[10:20, 48:52] = 0
    final = list(raw)
    if case == 'uncovered_ink':
        image[70:75, 48:52] = 0
    elif case == 'missing_child':
        final.remove(lower)
    elif case == 'missing_title':
        final.remove(raw[3])
    assert dedupe_visual_parents(raw, final, image) == final
    result = dedupe_visual_parents(raw, final, image, allow_mixed_children=True)
    assert result == ([b for b in final if b is not raw[0]] if case == 'complete' else final)


def test_visual_parent_is_restored_if_a_coverage_witness_loses_later_cleanup():
    from oxr.pipeline.orchestrator import _dedupe_full_page

    raw, image = visual_group(BlockType.PICTURE)
    stamp = block(38, BlockType.STAMP, [0, 90.5, 100, 100.5], 'caption', 0.99)
    raw.append(stamp)
    result = _dedupe_full_page(raw, list(raw), image)
    assert raw[3] not in result
    assert raw[0] in result
    assert stamp in result
