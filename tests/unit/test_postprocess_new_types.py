from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.chemical_structure_postprocess import postprocess_chemical_structure
from oxr.pipeline.postprocess.code_postprocess import postprocess_code


def test_chemical_structure_postprocess_outputs_somarkdown_smiles():
    block = Block(0, BlockType.CHEMICAL_STRUCTURE, [0, 0, 1, 1], "", ContentFormat.TEXT)

    result = postprocess_chemical_structure(block, "$\\smiles{CCO}$")

    assert result.content == "$\\smiles{CCO}$"
    assert result.format == ContentFormat.MARKDOWN


def test_code_postprocess_normalizes_markdown_code_block():
    block = Block(0, BlockType.CODE_BLOCK, [0, 0, 1, 1], "", ContentFormat.TEXT)

    result = postprocess_code(block, "```python\nprint('ok')\n```")

    assert result.content == "```\nprint('ok')\n```"
    assert result.format == ContentFormat.MARKDOWN
