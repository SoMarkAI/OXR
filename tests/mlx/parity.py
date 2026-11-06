"""Numerical parity check: MLX port vs the float32 PyTorch reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image

from oxr.model.mlx.preprocess import preprocess, resize_position_embedding
from tests.mlx.ref_torch import TorchOXR

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "OXR-1.0"


def _report(name: str, mine: np.ndarray, theirs: np.ndarray) -> float:
    """Print the numeric difference between two arrays."""
    diff = np.abs(mine.astype(np.float32) - theirs.astype(np.float32))
    scale = np.abs(theirs.astype(np.float32)).mean() + 1e-6
    print(f"{name:24s} max={diff.max():.3e}  mean={diff.mean():.3e}  rel={diff.max() / scale:.3e}")
    return float(diff.max())


def main() -> None:
    """Compare preprocessing, vision, visual tokens and logits."""
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--min-pixels", type=int, default=448 * 448)
    args = parser.parse_args()

    image = PIL.Image.open(args.image).convert("RGB")
    patches, grid_thw = preprocess(image, min_pixels=args.min_pixels)
    print(f"patches={patches.shape} grid_thw={grid_thw}")

    reference = TorchOXR(str(args.model_dir / "model.safetensors"))

    # Position embedding interpolation: numpy port vs torch.
    table = reference.w["visual_tokenizer.vit.vision_model.embeddings.position_embedding.weight"].numpy()
    mine_pe = resize_position_embedding(table, grid_thw[1], grid_thw[2])
    their_pe = reference.position_embedding(grid_thw[1], grid_thw[2]).numpy()
    _report("position embedding", mine_pe, their_pe)

    # Vision tower + visual vocabulary head.
    weights = mx.load(str(args.model_dir / "model.safetensors"))
    weights = {k: v.astype(mx.float32) for k, v in weights.items()}
    from oxr.model.mlx.model import OXR

    model = OXR(weights)
    mine_visual = np.array(model.visual_tokenizer.vit(patches, grid_thw))
    their_vision = reference.vision(patches, grid_thw).numpy()
    _report("vision features", mine_visual, their_vision)
    their_visual = (reference.visual_tokens(patches, grid_thw) @ reference.w["vte.weight"]).numpy()
    _report("visual embeds", np.array(model.image_embeds(patches, grid_thw)), their_visual)

    # Prompt logits.
    from oxr.model.mlx.runner import OXRRunner, render_chat

    runner = OXRRunner.__new__(OXRRunner)
    runner.tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(str(args.model_dir))
    runner.model = model
    ids, visual = runner.encode(image, "Recognize the content of the image.")
    mine_logits = np.array(model.llm(model.merge(ids, visual), None, 0))
    their_logits = reference.logits(ids, reference.visual_tokens(patches, grid_thw)).numpy()
    _report("logits", mine_logits, their_logits)
    print(f"top-1 mine={int(mine_logits[-1].argmax())} ref={int(their_logits[-1].argmax())}")
    print(f"prompt ids={len(ids)} tokens, rendered chat starts {render_chat('')[:20]!r}")


if __name__ == "__main__":
    main()
