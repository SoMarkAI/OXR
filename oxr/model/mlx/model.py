"""OXR-1.0 in MLX: SigLIP2-NaViT vision tower -> visual-token soft vocab -> Qwen3-0.6B.

Weights are read from the original Hugging Face checkpoint (no conversion step
required). The math mirrors the reference implementation used by the vLLM
plugin: windowed vision attention with 2D RoPE, a 65532-way visual vocabulary
followed by a LayerNorm, and the visual atoms injected as soft embeddings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import mlx.core as mx
import numpy as np

# Token ids used by the reference implementation.
IMAGE_PLACEHOLDER_ID = -200
VIDEO_PLACEHOLDER_ID = -201
VISUAL_ATOM_ID = -300
INDICATOR_IDS = (-301, -302, -303, -304)

EOS_IDS = (151643, 151645)


@dataclass(frozen=True)
class VisionConfig:
    hidden_size: int = 1152
    intermediate_size: int = 4304
    num_layers: int = 27
    num_heads: int = 16
    patch_size: int = 16
    hidden_stride: int = 2
    window_size: int = 112
    layer_norm_eps: float = 1e-6


@dataclass(frozen=True)
class TextConfig:
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_layers: int = 28
    num_heads: int = 16
    num_kv_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6


def _linear(x: mx.array, weight: mx.array, bias: mx.array | None = None) -> mx.array:
    """Apply a linear projection with an optional bias."""
    out = x @ weight.T
    return out if bias is None else out + bias


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float) -> mx.array:
    """Apply LayerNorm over the last axis."""
    return mx.fast.layer_norm(x, weight, bias, eps)


def _rms_norm(x: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Apply RMSNorm over the last axis."""
    return mx.fast.rms_norm(x, weight, eps)


def _rotate_half(x: mx.array) -> mx.array:
    """Rotate the two halves of the last axis, as in HF RoPE."""
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def _text_rope(x: mx.array, offset: int, head_dim: int, base: float) -> mx.array:
    """Half-split RoPE over the last axis of an (seq, heads, head_dim) tensor."""
    inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    positions = np.arange(x.shape[0], dtype=np.float32) + offset
    angles = np.outer(positions, inv_freq)
    cos = mx.array(np.concatenate([np.cos(angles), np.cos(angles)], axis=-1))[:, None, :]
    sin = mx.array(np.concatenate([np.sin(angles), np.sin(angles)], axis=-1))[:, None, :]
    return x * cos + _rotate_half(x) * sin


def _gelu_approx(x: mx.array) -> mx.array:
    """GELU with the tanh approximation used by SigLIP2."""
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


# --------------------------------------------------------------------------- #
# Vision: SigLIP2-NaViT
# --------------------------------------------------------------------------- #


def window_plan(
    grid_thw: tuple[int, int, int],
    config: VisionConfig,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Return (window_index, cu_window_seqlens, cu_seqlens) like the reference."""
    grid_t, grid_h, grid_w = grid_thw
    stride = config.hidden_stride
    spatial_merge_unit = stride * stride
    window_units = config.window_size // stride // config.patch_size

    llm_h, llm_w = grid_h // stride, grid_w // stride
    index = np.arange(grid_t * llm_h * llm_w).reshape(grid_t, llm_h, llm_w)
    pad_h = window_units - llm_h % window_units
    pad_w = window_units - llm_w % window_units
    num_windows_h = (llm_h + pad_h) // window_units
    num_windows_w = (llm_w + pad_w) // window_units
    padded = np.pad(index, ((0, 0), (0, pad_h), (0, pad_w)), constant_values=-100)
    padded = padded.reshape(grid_t, num_windows_h, window_units, num_windows_w, window_units)
    padded = padded.transpose(0, 1, 3, 2, 4).reshape(
        grid_t, num_windows_h * num_windows_w, window_units, window_units
    )
    seqlens = (padded != -100).sum(axis=(2, 3)).reshape(-1)
    flat = padded.reshape(-1)
    window_index = flat[flat != -100]

    cu_window = [0]
    cu_window.extend((np.cumsum(seqlens) * spatial_merge_unit + cu_window[0]).tolist())

    cu = np.repeat(np.array([grid_h * grid_w]), grid_t).cumsum()
    cu_seqlens = [0, *cu.tolist()]
    return window_index, cu_window, cu_seqlens


def _ropes(
    grid_thw: tuple[int, int, int],
    window_index: np.ndarray,
    config: VisionConfig,
) -> tuple[mx.array, mx.array]:
    """Per-patch 2D rotary frequencies, reordered into window order."""
    grid_t, grid_h, grid_w = grid_thw
    stride = config.hidden_stride
    head_dim = config.hidden_size // config.num_heads
    inv_freq = 1.0 / (10000.0 ** (np.arange(0, head_dim // 2, 2, dtype=np.float32) / (head_dim // 2)))

    h_ids = np.repeat(np.arange(grid_h), grid_w)
    w_ids = np.tile(np.arange(grid_w), grid_h)
    block = lambda ids: (
        ids.reshape(grid_h // stride, stride, grid_w // stride, stride)
        .transpose(0, 2, 1, 3)
        .reshape(-1)
    )
    h_block = np.tile(block(h_ids), grid_t)
    w_block = np.tile(block(w_ids), grid_t)

    freqs_h = h_block.astype(np.float32)[:, None] * inv_freq[None, :]
    freqs_w = w_block.astype(np.float32)[:, None] * inv_freq[None, :]
    rotary = np.concatenate([freqs_h, freqs_w], axis=-1)  # (patches, head_dim)

    unit = stride * stride
    rotary = rotary.reshape(len(rotary) // unit, unit, -1)[window_index].reshape(len(rotary), -1)
    # The reference builds cos/sin at 2x width and keeps the first half, which is
    # equivalent to duplicating them for the half-rotation form used below.
    angles = mx.array(np.concatenate([rotary, rotary], axis=-1))
    return mx.cos(angles), mx.sin(angles)


def _positional(
    weights: dict[str, mx.array],
    prefix: str,
    grid_thw: tuple[int, int, int],
) -> mx.array:
    """Interpolate the positional grid for this image and reshape it."""
    from .preprocess import resize_position_embedding

    _, grid_h, grid_w = grid_thw
    table = np.array(weights[f"{prefix}.embeddings.position_embedding.weight"].astype(mx.float32))
    resized = resize_position_embedding(table, grid_h, grid_w)
    return mx.array(resized)


def _segment_mask(cu_seqlens: Iterable[int]) -> mx.array:
    """Block-diagonal additive mask for variable-length segments."""
    bounds = list(cu_seqlens)
    length = bounds[-1]
    mask = np.full((length, length), -1e9, dtype=np.float32)
    for start, end in zip(bounds[:-1], bounds[1:]):
        mask[start:end, start:end] = 0.0
    return mx.array(mask)[None, None]


class VisionTower:
    """SigLIP2-NaViT encoder (27 windowed layers, 1152 hidden, 16 heads)."""

    def __init__(self, weights: dict[str, mx.array], prefix: str, config: VisionConfig):
        """Bind the vision weights for one prefix."""
        self.config = config
        self.prefix = prefix
        self.weights = weights
        patch = weights[f"{prefix}.embeddings.patch_embedding.weight"]
        self.patch_weight = patch.reshape(patch.shape[0], -1) if patch.ndim == 4 else patch
        self.patch_bias = weights.get(f"{prefix}.embeddings.patch_embedding.bias")

    def __call__(self, patches: np.ndarray, grid_thw: tuple[int, int, int]) -> mx.array:
        """Encode patch vectors into per-patch features."""
        config = self.config
        prefix = self.prefix
        weights = self.weights

        x = _linear(mx.array(patches), self.patch_weight, self.patch_bias)
        x = x + _positional(weights, prefix, grid_thw).astype(x.dtype)

        window_index, cu_window, cu_full = window_plan(grid_thw, config)
        unit = config.hidden_stride**2
        n_patches = x.shape[0]

        x = x.reshape(n_patches // unit, unit, -1)[mx.array(window_index)].reshape(n_patches, -1)
        cos, sin = _ropes(grid_thw, window_index, config)

        # fullatt_block_indexes is null in the released config, which makes every
        # layer use full (per-image) attention; the window permutation still applies.
        del cu_window
        mask = _segment_mask(cu_full)

        for index in range(config.num_layers):
            layer = f"{prefix}.encoder.layers.{index}"
            residual = x
            normed = _layer_norm(
                x,
                weights[f"{layer}.layer_norm1.weight"],
                weights[f"{layer}.layer_norm1.bias"],
                config.layer_norm_eps,
            )
            x = residual + self._attention(normed, layer, cos, sin, mask)

            residual = x
            normed = _layer_norm(
                x,
                weights[f"{layer}.layer_norm2.weight"],
                weights[f"{layer}.layer_norm2.bias"],
                config.layer_norm_eps,
            )
            hidden = _linear(normed, weights[f"{layer}.mlp.fc1.weight"], weights[f"{layer}.mlp.fc1.bias"])
            hidden = _gelu_approx(hidden)
            hidden = _linear(hidden, weights[f"{layer}.mlp.fc2.weight"], weights[f"{layer}.mlp.fc2.bias"])
            x = residual + hidden

        reverse = np.argsort(window_index)
        x = x.reshape(n_patches // unit, unit, -1)[mx.array(reverse)].reshape(n_patches, -1)
        return x

    def _attention(
        self,
        x: mx.array,
        layer: str,
        cos: mx.array,
        sin: mx.array,
        mask: mx.array,
    ) -> mx.array:
        """Windowed self-attention with 2D RoPE."""
        config = self.config
        weights = self.weights
        seq_len = x.shape[0]
        head_dim = config.hidden_size // config.num_heads

        q = _linear(x, weights[f"{layer}.self_attn.q_proj.weight"], weights[f"{layer}.self_attn.q_proj.bias"])
        k = _linear(x, weights[f"{layer}.self_attn.k_proj.weight"], weights[f"{layer}.self_attn.k_proj.bias"])
        v = _linear(x, weights[f"{layer}.self_attn.v_proj.weight"], weights[f"{layer}.self_attn.v_proj.bias"])

        q = q.reshape(seq_len, config.num_heads, head_dim)
        k = k.reshape(seq_len, config.num_heads, head_dim)
        v = v.reshape(seq_len, config.num_heads, head_dim)

        qf = q.astype(mx.float32)
        kf = k.astype(mx.float32)
        qr = qf * cos[:, None, :] + _rotate_half(qf) * sin[:, None, :]
        kr = kf * cos[:, None, :] + _rotate_half(kf) * sin[:, None, :]

        qh = qr.astype(x.dtype).transpose(1, 0, 2)[None]
        kh = kr.astype(x.dtype).transpose(1, 0, 2)[None]
        vh = v.transpose(1, 0, 2)[None]

        out = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=head_dim**-0.5, mask=mask
        )
        out = out[0].transpose(1, 0, 2).reshape(seq_len, config.hidden_size)
        return _linear(
            out,
            weights[f"{layer}.self_attn.out_proj.weight"],
            weights[f"{layer}.self_attn.out_proj.bias"],
        )


class VisualTokenizer:
    """Vision tower + visual-vocabulary head -> soft visual tokens."""

    def __init__(self, weights: dict[str, mx.array], prefix: str = "visual_tokenizer"):
        """Bind the tower and the visual-vocabulary head."""
        self.weights = weights
        self.prefix = prefix
        self.vit = VisionTower(weights, f"{prefix}.vit.vision_model", VisionConfig())
        self.visual_vocab_size = weights["vte.weight"].shape[0]

    def __call__(self, patches: np.ndarray, grid_thw: tuple[int, int, int]) -> mx.array:
        """Map an image to soft visual tokens."""
        features = self.vit(patches, grid_thw)
        unit = self.vit.config.hidden_stride**2
        features = features.reshape(features.shape[0] // unit, -1)

        head = f"{self.prefix}.head"
        logits = _linear(features, self.weights[f"{head}.0.weight"])
        # nn.LayerNorm default eps (1e-5) in the reference, computed in float32.
        logits = _layer_norm(
            logits.astype(mx.float32),
            self.weights[f"{head}.1.weight"].astype(mx.float32),
            self.weights[f"{head}.1.bias"].astype(mx.float32),
            1e-5,
        )
        probs = mx.softmax(logits, axis=-1).astype(features.dtype)
        pad = mx.zeros((probs.shape[0], len(INDICATOR_IDS)), dtype=probs.dtype)
        return mx.concatenate([probs, pad], axis=-1)


# --------------------------------------------------------------------------- #
# Text: Qwen3
# --------------------------------------------------------------------------- #


class Qwen3:
    def __init__(self, weights: dict[str, mx.array], prefix: str, config: TextConfig):
        """Bind the decoder weights for one prefix."""
        self.config = config
        self.prefix = prefix
        self.weights = weights
        self.embed = weights[f"{prefix}.model.embed_tokens.weight"]
        self.norm = weights[f"{prefix}.model.norm.weight"]
        self.lm_head = weights.get(f"{prefix}.lm_head.weight", self.embed)

    def __call__(self, inputs: mx.array, cache: list | None, offset: int) -> mx.array:
        """Run the decoder over embeddings and return next-token logits."""
        hidden = inputs
        for index in range(self.config.num_layers):
            hidden = self._layer(hidden, index, cache, offset)
        hidden = _rms_norm(hidden, self.norm, self.config.rms_norm_eps)
        return hidden @ self.lm_head.T

    def _layer(self, x: mx.array, index: int, cache: list | None, offset: int) -> mx.array:
        """One decoder block: attention, MLP and their residuals."""
        config = self.config
        base = f"{self.prefix}.model.layers.{index}"
        weights = self.weights
        seq_len = x.shape[0]
        head_dim = config.head_dim

        normed = _rms_norm(x, weights[f"{base}.input_layernorm.weight"], config.rms_norm_eps)
        q = _linear(normed, weights[f"{base}.self_attn.q_proj.weight"])
        k = _linear(normed, weights[f"{base}.self_attn.k_proj.weight"])
        v = _linear(normed, weights[f"{base}.self_attn.v_proj.weight"])

        q = q.reshape(seq_len, config.num_heads, head_dim)
        k = k.reshape(seq_len, config.num_kv_heads, head_dim)
        v = v.reshape(seq_len, config.num_kv_heads, head_dim)

        q = _rms_norm(q, weights[f"{base}.self_attn.q_norm.weight"], config.rms_norm_eps)
        k = _rms_norm(k, weights[f"{base}.self_attn.k_norm.weight"], config.rms_norm_eps)

        q = _text_rope(q, offset, head_dim, config.rope_theta)
        k = _text_rope(k, offset, head_dim, config.rope_theta)

        if cache is not None and cache[index] is not None:
            past_k, past_v = cache[index]
            k = mx.concatenate([past_k, k], axis=0)
            v = mx.concatenate([past_v, v], axis=0)
        if cache is not None:
            cache[index] = (k, v)

        qh = q.transpose(1, 0, 2)[None]
        kh = k.transpose(1, 0, 2)
        vh = v.transpose(1, 0, 2)
        groups = config.num_heads // config.num_kv_heads
        kh = mx.repeat(kh, groups, axis=0)[None]
        vh = mx.repeat(vh, groups, axis=0)[None]

        mask = None
        if seq_len > 1:
            total = k.shape[0]
            causal = np.full((seq_len, total), -1e9, dtype=np.float32)
            causal[np.tril_indices(seq_len, k=0, m=total)] = 0.0
            mask = mx.array(causal)[None, None]

        out = mx.fast.scaled_dot_product_attention(
            qh, kh, vh, scale=head_dim**-0.5, mask=mask
        )
        out = out[0].transpose(1, 0, 2).reshape(seq_len, config.num_heads * config.head_dim)
        x = x + _linear(out, weights[f"{base}.self_attn.o_proj.weight"])

        normed = _rms_norm(x, weights[f"{base}.post_attention_layernorm.weight"], config.rms_norm_eps)
        gate = _linear(normed, weights[f"{base}.mlp.gate_proj.weight"])
        up = _linear(normed, weights[f"{base}.mlp.up_proj.weight"])
        act = mx.sigmoid(gate) * gate * up
        return x + _linear(act, weights[f"{base}.mlp.down_proj.weight"])


class OXR:
    """Full model: image -> soft visual tokens -> Qwen3 embeddings -> next-token logits."""

    def __init__(self, weights: dict[str, mx.array]):
        """Assemble the visual tokenizer, decoder and embedding table."""
        self.weights = weights
        self.visual_tokenizer = VisualTokenizer(weights)
        self.llm = Qwen3(weights, "llm", TextConfig())
        self.vte = weights["vte.weight"]
        # Indicator ids are negative placeholders; their rows sit at the tail of the
        # visual vocabulary (visual_vocab_size - 4 .. visual_vocab_size - 1).
        tail = self.vte.shape[0] - len(INDICATOR_IDS)
        self.indicator_embeds = self.vte[mx.array(list(range(tail, self.vte.shape[0])))]

    def image_embeds(self, patches: np.ndarray, grid_thw: tuple[int, int, int]) -> mx.array:
        """Project an image's soft tokens into the LM embedding space."""
        tokens = self.visual_tokenizer(patches, grid_thw)
        return tokens @ self.vte

    def merge(self, ids: np.ndarray, visual: mx.array | None) -> mx.array:
        """Build the prompt embedding sequence, splicing visual atoms in place."""
        tokens = [int(token) for token in ids]
        parts: list[mx.array] = []
        text_buffer: list[int] = []
        atom = 0

        def flush_text() -> None:
            """Emit buffered text tokens as embeddings."""
            if text_buffer:
                parts.append(self.llm.embed[mx.array(np.array(text_buffer, dtype=np.int32))])
                text_buffer.clear()

        index = 0
        while index < len(tokens):
            token = tokens[index]
            if token == VISUAL_ATOM_ID:
                flush_text()
                end = index
                while end < len(tokens) and tokens[end] == VISUAL_ATOM_ID:
                    end += 1
                if visual is None:
                    raise ValueError("prompt contains visual atoms but no image was provided")
                parts.append(visual[atom : atom + (end - index)])
                atom += end - index
                index = end
                continue
            if token < 0:
                flush_text()
                parts.append(self.indicator_embeds[INDICATOR_IDS.index(token)][None])
            else:
                text_buffer.append(token)
            index += 1
        flush_text()

        if visual is not None and atom != visual.shape[0]:
            raise ValueError(f"used {atom} visual atoms but the image produced {visual.shape[0]}")
        return mx.concatenate(parts, axis=0)
