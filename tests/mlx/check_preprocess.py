"""Check the numpy preprocessing against transformers' SiglipImageProcessor."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import PIL.Image
from transformers import AutoImageProcessor

from oxr.model.mlx.preprocess import preprocess

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "OXR-1.0"

MODEL = str(DEFAULT_MODEL_DIR)


def hf_preprocess(image: PIL.Image.Image, min_pixels: int = 448 * 448, max_pixels: int = 921600):
    """Verbatim port of OXRVisualTokenizer.preprocess from the upstream model code."""
    processor = AutoImageProcessor.from_pretrained(
        MODEL, do_center_crop=False, backend="pil"
    )
    patch_size, temporal_patch_size, hidden_stride = 16, 1, 2
    images = [image.convert("RGB")]
    width, height = images[0].size
    processed = []
    for item in images:
        from oxr.model.mlx.preprocess import smart_resize

        resized_height, resized_width = smart_resize(
            height, width, factor=patch_size * hidden_stride, min_pixels=min_pixels, max_pixels=max_pixels
        )
        new_image = processor.preprocess(
            item, size=dict(height=resized_height, width=resized_width), return_tensors="np"
        )["pixel_values"][0]
        processed.append(new_image)
    patches = np.array(processed)
    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
    grid_thw = np.array([[grid_t, grid_h, grid_w]])
    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // hidden_stride,
        hidden_stride,
        patch_size,
        grid_w // hidden_stride,
        hidden_stride,
        patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flat = patches.reshape(grid_t * grid_h * grid_w, channel * temporal_patch_size * patch_size * patch_size)
    return flat, tuple(grid_thw[0])


def main() -> None:
    """Diff the numpy patchification against transformers."""
    image = PIL.Image.open(sys.argv[1]).convert("RGB")
    mine, my_grid = preprocess(image)
    theirs, their_grid = hf_preprocess(image)
    print("grid", my_grid, their_grid)
    print("shape", mine.shape, theirs.shape)
    diff = np.abs(mine.astype(np.float32) - theirs.astype(np.float32))
    print(f"patch diff max={diff.max():.6f} mean={diff.mean():.6f}")
    if diff.max() > 1e-5:
        np.save("/tmp/patches_mine.npy", mine)
        np.save("/tmp/patches_hf.npy", theirs)


if __name__ == "__main__":
    main()
