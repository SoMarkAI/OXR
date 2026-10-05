import pytest
from oxr.pipeline.models import Block, BlockType, ContentFormat
from oxr.pipeline.postprocess.assembly_postprocess import blocks_to_markdown

def test_blocks_to_markdown():
    blocks = [
        Block(idx=0, type=BlockType.TITLE, bbox=[0,0,10,10], content="Title", format=ContentFormat.TEXT),
        Block(idx=1, type=BlockType.TEXT, bbox=[0,10,10,20], content="Hello world", format=ContentFormat.TEXT),
        Block(idx=2, type=BlockType.CODE_BLOCK, bbox=[0,20,10,30], content="```\nprint('ok')\n```", format=ContentFormat.MARKDOWN),
        Block(idx=3, type=BlockType.CHEMICAL_STRUCTURE, bbox=[0,30,10,40], content="$\\smiles{CCO}$", format=ContentFormat.MARKDOWN),
    ]
    md = blocks_to_markdown(blocks)
    assert "# Title" in md
    assert "Hello world" in md
    assert "print('ok')" in md
    assert "$\\smiles{CCO}$" in md
