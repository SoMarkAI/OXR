import asyncio

import numpy as np
from oxr.config.settings import settings
from oxr.model.client import ModelOutput
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.recognition import recognize_block


def _block(block_type: BlockType) -> Block:
    return Block(0, block_type, [0, 0, 4, 4], "", ContentFormat.TEXT)


def test_recognition_routes_by_block_type(monkeypatch, tmp_path):
    calls = []

    async def record(name):
        async def inner(_image, _model_options=None):
            calls.append(name)
            return {
                "text": "hello",
                "table": "<table></table>",
                "formula": "x^2",
                "code": "print('ok')",
                "chemical": "CCO",
            }[name]
        return inner

    monkeypatch.setattr(settings.pipeline, "image_output_dir", str(tmp_path))
    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_text", asyncio.run(record("text")))
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_table", asyncio.run(record("table")))
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_formula", asyncio.run(record("formula")))
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_code", asyncio.run(record("code")))
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_chemical_structure",
        asyncio.run(record("chemical")),
    )

    async def scenario():
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        sem = asyncio.Semaphore(8)
        results = []
        for block_type in [
            BlockType.TEXT,
            BlockType.STAMP,
            BlockType.CHEMICAL_STRUCTURE,
            BlockType.TABLE,
            BlockType.FORMULA,
            BlockType.CODE_BLOCK,
            BlockType.PICTURE,
        ]:
            results.append(await recognize_block(_block(block_type), image, sem, "demo.png", 0))
        return results

    results = asyncio.run(scenario())

    assert calls == ["text", "text", "chemical", "table", "formula", "code"]
    assert results[2].content == "$\\smiles{CCO}$"
    assert results[3].format == ContentFormat.HTML
    assert results[4].format == ContentFormat.LATEX
    assert results[5].format == ContentFormat.MARKDOWN
    assert results[6].format == ContentFormat.MARKDOWN


def test_recognition_outputs_table_markdown_when_requested(monkeypatch):
    async def recognize_table(_image, _model_options=None):
        return "<table><tr><th>A</th></tr><tr><td>B</td></tr></table>"

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_table", recognize_table)

    async def scenario():
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        sem = asyncio.Semaphore(8)
        return await recognize_block(
            _block(BlockType.TABLE),
            image,
            sem,
            "demo.png",
            0,
            element_formats={"table": "markdown"},
        )

    result = asyncio.run(scenario())

    assert result.format == ContentFormat.MARKDOWN
    assert result.content == "| A |\n| --- |\n| B |"


def test_recognition_routes_chemical_structure_smiles(monkeypatch):
    async def recognize_chemical_structure(_image, _model_options=None):
        return "CCO"

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_chemical_structure",
        recognize_chemical_structure,
    )

    async def scenario():
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        sem = asyncio.Semaphore(8)
        return await recognize_block(
            _block(BlockType.CHEMICAL_STRUCTURE),
            image,
            sem,
            "demo.png",
            0,
            element_formats={"cs": "smiles"},
        )

    result = asyncio.run(scenario())

    assert result.format == ContentFormat.MARKDOWN
    assert result.content == "$\\smiles{CCO}$"


def test_recognition_routes_chemical_structure_image(monkeypatch, tmp_path):
    calls = []

    async def recognize_text(_image, _model_options=None):
        calls.append("text")
        return "CCO"

    monkeypatch.setattr(settings.pipeline, "image_output_dir", str(tmp_path))
    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr("oxr.pipeline.recognition.model_client.recognize_text", recognize_text)

    async def scenario():
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        sem = asyncio.Semaphore(8)
        return await recognize_block(
            _block(BlockType.CHEMICAL_STRUCTURE),
            image,
            sem,
            "demo.png",
            0,
            element_formats={"cs": "image"},
        )

    result = asyncio.run(scenario())

    assert calls == []
    assert result.format == ContentFormat.MARKDOWN
    assert result.content.startswith("![](")
    assert (tmp_path / "demo" / "imgs" / "0_0.png").exists()


def test_text_length_response_retries_with_repetition_penalty(monkeypatch):
    calls = []

    async def recognize_text(image, model_options=None):
        calls.append((image, model_options))
        if len(calls) == 1:
            return ModelOutput("looping", finish_reason="length")
        return ModelOutput("recovered", finish_reason="stop")

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_text",
        recognize_text,
    )

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = asyncio.run(
        recognize_block(
            _block(BlockType.TEXT),
            image,
            asyncio.Semaphore(1),
            "demo.png",
            0,
            model_options={"temperature": 0.0},
        )
    )

    assert result.content == "recovered"
    assert len(calls) == 2
    assert calls[0][0].shape == (4, 4, 3)
    assert calls[1][0] is calls[0][0]
    assert calls[0][1] == {"temperature": 0.0}
    assert calls[1][1] == {
        "temperature": 0.0,
        "repetition_penalty": 1.1,
    }


def test_table_length_response_retries_with_downscaled_crop(monkeypatch):
    calls = []

    async def recognize_table(image, model_options=None):
        calls.append((image, model_options))
        if len(calls) == 1:
            return ModelOutput("<table>", finish_reason="length")
        return ModelOutput("<table></table>", finish_reason="stop")

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_table",
        recognize_table,
    )

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    result = asyncio.run(
        recognize_block(
            _block(BlockType.TABLE),
            image,
            asyncio.Semaphore(1),
            "demo.png",
            0,
        )
    )

    assert result.content == "<table></table>"
    assert [call[0].shape for call in calls] == [(4, 4, 3), (3, 3, 3)]
    assert calls[0][1] is None
    assert calls[1][1] is None


def test_unrecovered_text_length_response_preserves_model_output(monkeypatch):
    calls = []

    async def recognize_text(_image, _model_options=None):
        calls.append(_model_options)
        return ModelOutput("looping", finish_reason="length")

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_text",
        recognize_text,
    )

    result = asyncio.run(
        recognize_block(
            _block(BlockType.TEXT),
            np.zeros((8, 8, 3), dtype=np.uint8),
            asyncio.Semaphore(1),
            "demo.png",
            0,
        )
    )

    assert result.content == "looping"
    assert calls == [None, {"repetition_penalty": 1.1}]


def test_formula_length_response_preserves_model_output_without_retry(monkeypatch):
    calls = []

    async def recognize_formula(_image, _model_options=None):
        calls.append(_model_options)
        return ModelOutput("x^2", finish_reason="length")

    monkeypatch.setattr(settings.oxr_model, "retry_times", 1)
    monkeypatch.setattr(
        "oxr.pipeline.recognition.model_client.recognize_formula",
        recognize_formula,
    )

    result = asyncio.run(
        recognize_block(
            _block(BlockType.FORMULA),
            np.zeros((8, 8, 3), dtype=np.uint8),
            asyncio.Semaphore(1),
            "demo.png",
            0,
        )
    )

    assert result.content == "x^2"
    assert calls == [None]
