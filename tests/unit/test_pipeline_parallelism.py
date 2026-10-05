import asyncio

import numpy as np
import pytest

from oxr.config.settings import settings
from oxr.model.client import model_client
from oxr.pipeline import orchestrator
from oxr.pipeline.concurrency import gather_cancel_on_error
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.reading_order import resolve_reading_orders
from oxr.pipeline.recognition import recognize_block, recognize_pages


def _block(idx: int, block_type: BlockType = BlockType.TEXT, bbox=None) -> Block:
    return Block(idx, block_type, bbox or [0, 0, 4, 4], "", ContentFormat.TEXT)


def test_gather_cancel_on_error_cancels_and_drains_siblings():
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def fail_after_sibling_starts():
        await sibling_started.wait()
        raise RuntimeError("branch failed")

    async def sibling():
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()

    async def scenario():
        with pytest.raises(RuntimeError, match="branch failed"):
            await gather_cancel_on_error(fail_after_sibling_starts(), sibling())
        assert sibling_cancelled.is_set()

    asyncio.run(scenario())


def test_reading_order_and_recognition_share_combined_oxr_concurrency(monkeypatch):
    active = 0
    maximum = 0
    two_requests_entered = asyncio.Event()
    release = asyncio.Event()

    async def enter_request():
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        if active == 2:
            two_requests_entered.set()
        await release.wait()
        active -= 1

    async def request_reading_order(_prompt, _image, _max_tokens):
        await enter_request()
        return "0 1"

    async def recognize_text(_image, _model_options=None):
        await enter_request()
        return "text"

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr("oxr.pipeline.reading_order.model_client.request_reading_order", request_reading_order)
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_text", recognize_text)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    pages = [[_block(0, bbox=[0, 0, 4, 4]), _block(1, bbox=[4, 4, 8, 8])]]

    async def scenario():
        semaphore = asyncio.Semaphore(2)
        order_task = asyncio.create_task(resolve_reading_orders(pages, [image], semaphore))
        recognition_task = asyncio.create_task(
            recognize_pages(pages, [image], "demo.png", semaphore)
        )
        await asyncio.wait_for(two_requests_entered.wait(), timeout=1)
        release.set()
        await gather_cancel_on_error(order_task, recognition_task)

    asyncio.run(scenario())

    assert maximum == 2


def test_concurrent_documents_share_process_oxr_concurrency_limit(monkeypatch):
    active = 0
    maximum = 0

    async def analyze_layouts(_images, _model_options=None):
        return [[_block(0)]]

    async def recognize(
        pages_blocks,
        _images,
        _file_name,
        semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        nonlocal active, maximum
        async with semaphore:
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.01)
            active -= 1
        return pages_blocks

    monkeypatch.setattr(
        orchestrator.cv2,
        "imdecode",
        lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", True)
    monkeypatch.setattr(settings.oxr_model, "max_concurrency", 1)

    async def scenario():
        await asyncio.gather(
            orchestrator.process_document(b"first", "first.png", ["json"]),
            orchestrator.process_document(b"second", "second.png", ["json"]),
        )

    asyncio.run(scenario())

    assert maximum == 1


def test_process_oxr_limiter_is_safe_across_asyncio_run_loops(monkeypatch):
    monkeypatch.setattr(settings.oxr_model, "max_concurrency", 1)

    async def bind_limiter_to_current_loop():
        semaphore = model_client.get_request_semaphore()
        await semaphore.acquire()
        waiter = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0)
        assert not waiter.done()
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        semaphore.release()
        return semaphore

    first = asyncio.run(bind_limiter_to_current_loop())
    second = asyncio.run(bind_limiter_to_current_loop())

    assert first is not second


def test_recognition_releases_semaphore_during_retry_backoff(monkeypatch):
    call_count = 0
    second_request_finished = asyncio.Event()

    async def recognize_text(_image, _model_options=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("first attempt fails")
        if call_count == 2:
            second_request_finished.set()
        return "ok"

    async def retry_backoff(_seconds):
        await asyncio.wait_for(second_request_finished.wait(), timeout=1)

    monkeypatch.setattr(settings.oxr_model, "retry_times", 2)
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_text", recognize_text)
    monkeypatch.setattr("oxr.pipeline.recognition.asyncio.sleep", retry_backoff)

    async def scenario():
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        semaphore = asyncio.Semaphore(1)
        await gather_cancel_on_error(
            recognize_block(_block(0), image, semaphore, "demo.png", 0),
            recognize_block(_block(1), image, semaphore, "demo.png", 0),
        )

    asyncio.run(scenario())

    assert call_count == 3


def test_recognize_pages_preserves_input_positions_without_sorting(monkeypatch):
    async def recognize_text(image, _model_options=None):
        return str(int(image[0, 0, 0]))

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_text", recognize_text)

    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, 4:, :] = 7
    blocks = [_block(5, bbox=[0, 0, 4, 4]), _block(1, bbox=[4, 0, 8, 4])]

    result = asyncio.run(
        recognize_pages([blocks], [image], "demo.png", asyncio.Semaphore(2))
    )

    assert [block.idx for block in result[0]] == [5, 1]
    assert [block.content for block in result[0]] == ["0", "7"]


def test_orchestrator_starts_reading_order_before_recognition_and_overlaps_branches(monkeypatch):
    starts = []
    reading_order_started = asyncio.Event()
    recognition_started = asyncio.Event()

    async def analyze_layouts(_images, _model_options=None):
        return [[_block(0), _block(1)]]

    async def resolve_orders(pages_blocks, _images, _semaphore):
        starts.append("reading-order")
        reading_order_started.set()
        await recognition_started.wait()
        return [list(range(len(page))) for page in pages_blocks]

    async def recognize(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        starts.append("recognition")
        recognition_started.set()
        await reading_order_started.wait()
        return pages_blocks

    monkeypatch.setattr(orchestrator.cv2, "imdecode", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", resolve_orders)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", False)

    asyncio.run(orchestrator.process_document(b"image", "page.png", ["json"], keep_header_footer=True))

    assert starts == ["reading-order", "recognition"]


def test_orchestrator_bypasses_reading_order_and_preserves_layout_order(monkeypatch):
    async def analyze_layouts(_images, _model_options=None):
        return [[_block(0, BlockType.TITLE), _block(1, BlockType.TEXT)]]

    async def unexpected_resolve(*_args, **_kwargs):
        raise AssertionError("reading order must be bypassed")

    async def recognize(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        return pages_blocks

    monkeypatch.setattr(orchestrator.cv2, "imdecode", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", unexpected_resolve)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", True)

    result = asyncio.run(
        orchestrator.process_document(b"image", "page.png", ["json"], keep_header_footer=True)
    )

    assert [block["type"] for block in result["outputs"]["json"]["pages"][0]["blocks"]] == [
        BlockType.TITLE.value,
        BlockType.TEXT.value,
    ]


def test_orchestrator_applies_local_permutation_then_expands_formula_indexes(monkeypatch):
    async def analyze_layouts(_images, _model_options=None):
        return [[
            _block(0, BlockType.TEXT),
            _block(1, BlockType.TEXT),
            _block(2, BlockType.FORMULA),
        ]]

    async def resolve_orders(_pages_blocks, _images, _semaphore):
        return [[2, 0, 1]]

    async def recognize(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        pages_blocks[0][0].content = "first"
        pages_blocks[0][1].content = "second"
        pages_blocks[0][2].content = "x"
        pages_blocks[0][2].format = ContentFormat.LATEX
        pages_blocks[0][2].text_after = "(1)"
        return pages_blocks

    monkeypatch.setattr(orchestrator.cv2, "imdecode", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", resolve_orders)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", False)

    result = asyncio.run(
        orchestrator.process_document(b"image", "page.png", ["json"], keep_header_footer=True)
    )
    blocks = result["outputs"]["json"]["pages"][0]["blocks"]

    assert [block["idx"] for block in blocks] == [0, 2, 3]
    assert [(block["type"], block["content"]) for block in blocks] == [
        (BlockType.FORMULA.value, r"x\tag{1}"),
        (BlockType.TEXT.value, "first"),
        (BlockType.TEXT.value, "second"),
    ]
    assert all("generated_role" not in block for block in blocks)


def test_orchestrator_dedupes_before_expanding_formula_numbering(monkeypatch):
    async def analyze_layouts(_images, _model_options=None):
        blocks = [
            _block(0, BlockType.TEXT, [0, 0, 4, 4]),
            _block(1, BlockType.FORMULA, [0, 0, 4, 4]),
            _block(2, BlockType.TEXT, [0, 4, 4, 8]),
            _block(3, BlockType.FORMULA, [0, 4, 4, 8]),
        ]
        for block, score in zip(blocks, [0.9, 0.2, 0.3, 0.8]):
            block.layout_score = score
        return [blocks]

    async def recognize(
        pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        contents = ["$x$", "x", r"\[y\]", "y"]
        for block, content in zip(pages_blocks[0], contents):
            block.content = content
        pages_blocks[0][1].format = ContentFormat.LATEX
        pages_blocks[0][1].text_after = "removed numbering"
        pages_blocks[0][3].format = ContentFormat.LATEX
        pages_blocks[0][3].text_after = "(2)"
        return pages_blocks

    monkeypatch.setattr(orchestrator.cv2, "imdecode", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", True)

    result = asyncio.run(
        orchestrator.process_document(b"image", "page.png", ["json"], keep_header_footer=True)
    )
    blocks = result["outputs"]["json"]["pages"][0]["blocks"]

    assert [block["idx"] for block in blocks] == [0, 1]
    assert [(block["type"], block["content"]) for block in blocks] == [
        (BlockType.TEXT.value, "$x$"),
        (BlockType.FORMULA.value, r"y\tag{2}"),
    ]


@pytest.mark.parametrize("provides_reading_order", [True, False])
def test_orchestrator_exact_text_dedupe_only_changes_final_outputs(monkeypatch, provides_reading_order):
    blocks = [
        Block(8, BlockType.TEXT, [0, 0, 4, 4], "", ContentFormat.TEXT, layout_score=0.6),
        Block(3, BlockType.CAPTION, [0, 0, 4, 4], "", ContentFormat.TEXT, layout_score=0.7),
        Block(5, BlockType.TITLE, [0, 0, 4, 4], "", ContentFormat.TEXT, layout_score=0.9),
    ]

    async def analyze_layouts(_images, _model_options=None):
        return [blocks]

    async def resolve_orders(pages, _images, _semaphore):
        assert len(pages[0]) == 3
        assert all(a is b for a, b in zip(pages[0], blocks))
        return [[0, 1, 2]]

    async def recognize(pages, *_args):
        assert len(pages[0]) == 3
        assert all(a is b for a, b in zip(pages[0], blocks))
        for block in pages[0]:
            block.content = "same heading"
        return pages

    monkeypatch.setattr(orchestrator.cv2, "imdecode", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", resolve_orders)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", provides_reading_order)

    result = asyncio.run(orchestrator.process_document(b"image", "page.png", ["json", "markdown"]))
    final = result["outputs"]["json"]["pages"][0]["blocks"]
    # Formula-number expansion already assigns sequential idx before this rule.
    assert final == [{"idx": 2, "type": "Title", "bbox": [0, 0, 4, 4], "content": "same heading", "format": "text"}]
    assert result["outputs"]["markdown"].count("same heading") == 1
    assert len(blocks) == 3
    assert [block.idx for block in blocks] == [0, 1, 2]


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("provides_reading_order", [True, False])
def test_full_overlap_mode_switches_final_policy_without_changing_model_inputs(
    monkeypatch, enabled, provides_reading_order,
):
    blocks = [
        Block(0, BlockType.TEXT, [0, 0, 40, 20], "", ContentFormat.TEXT, layout_score=0.4),
        Block(1, BlockType.FORMULA, [2, 2, 12, 12], "", ContentFormat.LATEX, layout_score=0.95),
        Block(2, BlockType.TEXT, [50, 0, 70, 20], "", ContentFormat.TEXT, layout_score=0.2),
        Block(3, BlockType.FORMULA, [60, 0, 80, 20], "", ContentFormat.LATEX, layout_score=0.9),
        Block(4, BlockType.PICTURE, [90, 0, 120, 20], "", ContentFormat.TEXT, layout_score=0.8),
        Block(5, BlockType.CAPTION, [95, 5, 115, 15], "", ContentFormat.TEXT, layout_score=0.99),
    ]

    async def analyze_layouts(_images, _model_options=None):
        return [blocks]

    async def resolve_orders(pages, _images, _semaphore):
        assert len(pages[0]) == 6
        return [list(range(6))]

    async def recognize(pages, *_args):
        assert len(pages[0]) == 6
        for block, content in zip(pages[0], ["paragraph", "x", "text", "y", "picture", "caption"]):
            block.content = content
        pages[0][3].text_after = "(2)"
        return pages

    def unexpected_local_policy(*_args):
        pytest.fail("full mode must replace the local cleanup policies")

    if enabled:
        for name in (
            "dedupe_systematic_cross_category_blocks",
            "dedupe_complete_text", "dedupe_exact_textual_blocks",
        ):
            monkeypatch.setattr(orchestrator, name, unexpected_local_policy)
    monkeypatch.setattr(settings.pipeline, "full_overlap_dedupe", enabled)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", provides_reading_order)
    monkeypatch.setattr(orchestrator.cv2, "imdecode", lambda *_args: np.zeros((24, 128, 3), dtype=np.uint8))
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", resolve_orders)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)

    result = asyncio.run(orchestrator.process_document(b"image", "page.png", ["json", "markdown"]))
    final = result["outputs"]["json"]["pages"][0]["blocks"]
    # Full mode absorbs the contained formula despite different OCR content;
    # merely intersecting rows and generated equation numbers still survive.
    expected = [0, 2, 3, 5, 6] if enabled else [0, 1, 2, 3, 5, 6]
    assert [block["idx"] for block in final] == expected
    expected_content = ["paragraph", "text", r"y\tag{2}", "picture", "caption"] if enabled else [
        "paragraph", "x", "text", r"y\tag{2}", "picture", "caption",
    ]
    assert [block["content"] for block in final] == expected_content
    markdown = result["outputs"]["markdown"]
    assert "caption" in markdown
    assert "(2)" in markdown  # Generated numbering has no LA confidence to compare.
    assert len(blocks) == 6


def test_orchestrator_cancels_recognition_when_reading_order_fails(monkeypatch):
    recognition_started = asyncio.Event()
    recognition_cancelled = asyncio.Event()

    async def analyze_layouts(_images, _model_options=None):
        return [[_block(0), _block(1)]]

    async def fail_reading_order(_pages_blocks, _images, _semaphore):
        await recognition_started.wait()
        raise RuntimeError("reading-order failed")

    async def recognize(
        _pages_blocks,
        _images,
        _file_name,
        _semaphore,
        _model_options=None,
        _element_formats=None,
    ):
        recognition_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            recognition_cancelled.set()

    monkeypatch.setattr(orchestrator.cv2, "imdecode", lambda *_args: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(orchestrator, "analyze_layouts", analyze_layouts)
    monkeypatch.setattr(orchestrator, "resolve_reading_orders", fail_reading_order)
    monkeypatch.setattr(orchestrator, "recognize_pages", recognize)
    monkeypatch.setattr(settings.layout_analyze_model, "provides_reading_order", False)

    with pytest.raises(RuntimeError, match="reading-order failed"):
        asyncio.run(
            orchestrator.process_document(b"image", "page.png", ["json"], keep_header_footer=True)
        )

    assert recognition_cancelled.is_set()
