"""Image preprocessing for OXR-1.0 (SigLIP2-NaViT patchification).

Mirrors the production (vLLM plugin) path:
  smart_resize -> bicubic resize -> rescale/normalize -> 2x2 block patchify
so the vectors fed to the vision tower match the served model.
"""

from __future__ import annotations

import math

import numpy as np
import PIL.Image

PATCH_SIZE = 16
HIDDEN_STRIDE = 2
TEMPORAL_PATCH_SIZE = 1
FACTOR = PATCH_SIZE * HIDDEN_STRIDE  # 32
MIN_PIXELS = 448 * 448
MAX_PIXELS = 921600

_MEAN = 0.5
_STD = 0.5
_RESCALE = 1.0 / 255.0


def smart_resize(
    height: int,
    width: int,
    factor: int = FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Resize so both sides are multiples of `factor` and the pixel budget holds."""
    if height < factor or width < factor:
        if height < width:
            width = round(factor / height * width)
            height = factor
        else:
            height = round(factor / width * height)
            width = factor
    elif max(height, width) / min(height, width) > 200:
        if height > width:
            height = 200 * width
        else:
            width = 200 * height

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _to_chw_normalized(image: PIL.Image.Image) -> np.ndarray:
    """Convert to CHW and apply the SigLIP rescale and normalisation."""
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
    return (array * _RESCALE - _MEAN) / _STD


def patchify(chw: np.ndarray) -> np.ndarray:
    """(3, H, W) -> (h*w, 3*16*16) with the 2x2 patch blocks kept contiguous."""
    _, height, width = chw.shape
    grid_h, grid_w = height // PATCH_SIZE, width // PATCH_SIZE
    blocks = chw.reshape(
        3,
        grid_h // HIDDEN_STRIDE,
        HIDDEN_STRIDE,
        PATCH_SIZE,
        grid_w // HIDDEN_STRIDE,
        HIDDEN_STRIDE,
        PATCH_SIZE,
    )
    blocks = blocks.transpose(1, 4, 2, 5, 0, 3, 6)
    return blocks.reshape(grid_h * grid_w, 3 * PATCH_SIZE * PATCH_SIZE)


def preprocess(
    image: PIL.Image.Image,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Return (patches [n_patches, 768], grid_thw [t, h, w] in patches)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size
    resized_h, resized_w = smart_resize(height, width, min_pixels=min_pixels, max_pixels=max_pixels)
    resized = image.resize((resized_w, resized_h), resample=PIL.Image.BICUBIC)
    chw = _to_chw_normalized(resized)
    grid_thw = (1, resized_h // PATCH_SIZE, resized_w // PATCH_SIZE)
    return patchify(chw), grid_thw


def _cubic_weight(x: np.ndarray, a: float = -0.75) -> np.ndarray:
    """Cubic convolution kernel used by bicubic resize (a = -0.75)."""
    abs_x = np.abs(x)
    near = (a + 2.0) * abs_x**3 - (a + 3.0) * abs_x**2 + 1.0
    far = a * abs_x**3 - 5.0 * a * abs_x**2 + 8.0 * a * abs_x - 4.0 * a
    return np.where(abs_x <= 1.0, near, np.where(abs_x < 2.0, far, 0.0))


def _resize_axis(source: np.ndarray, out_size: int, axis: int) -> np.ndarray:
    """Bicubic resize along one axis, matching torch F.interpolate(align_corners=False)."""
    source = np.moveaxis(source, axis, -1).astype(np.float32)
    in_size = source.shape[-1]
    scale = in_size / out_size
    dst = np.arange(out_size, dtype=np.float32)
    real = scale * (dst + 0.5) - 0.5
    # torch only clamps negative source indices for non-cubic modes
    base = np.floor(real).astype(np.int64)
    frac = real - base
    offsets = np.stack([base - 1, base, base + 1, base + 2], axis=-1)
    offsets = np.clip(offsets, 0, in_size - 1)
    distances = np.stack([frac + 1.0, frac, 1.0 - frac, 2.0 - frac], axis=-1)
    weights = _cubic_weight(distances)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    gathered = source[..., offsets]
    out = (gathered * weights).sum(axis=-1)
    return np.moveaxis(out, -1, axis)


def resize_position_embedding(
    weights: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Interpolate the (32, 32, dim) positional grid to (out_h, out_w) then reorder blocks."""
    grid = int(round(math.sqrt(weights.shape[0])))
    dim = weights.shape[1]
    positional = weights.reshape(grid, grid, dim).transpose(2, 0, 1)  # (dim, g, g)
    positional = _resize_axis(positional, out_h, axis=1)
    positional = _resize_axis(positional, out_w, axis=2)
    patches = positional.transpose(1, 2, 0)  # (out_h, out_w, dim)
    patches = patches.reshape(
        out_h // HIDDEN_STRIDE,
        HIDDEN_STRIDE,
        out_w // HIDDEN_STRIDE,
        HIDDEN_STRIDE,
        dim,
    )
    patches = patches.transpose(0, 2, 1, 3, 4)
    return patches.reshape(out_h * out_w, dim)
