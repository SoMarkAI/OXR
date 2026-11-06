"""PyTorch (CPU, float32) reference for OXR-1.0, written from the upstream math.

Used only to check the MLX port numerically: it takes the same patch vectors the
MLX pipeline produces and returns vision features / logits.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

DTYPE = torch.float32


class TorchOXR:
    def __init__(self, model_path: str):
        """Load the checkpoint in float32 on CPU."""
        self.w = {k: v.to(DTYPE) for k, v in load_file(model_path).items()}
        self.hidden_size = 1152
        self.heads = 16
        self.layers = 27
        self.hidden_stride = 2
        self.patch_size = 16
        self.window_size = 112
        self.eps = 1e-6
        self.merge_unit = self.hidden_stride**2

    # ---------------- vision ----------------
    def position_embedding(self, grid_h: int, grid_w: int) -> torch.Tensor:
        """Interpolate the positional grid like the reference."""
        table = self.w["visual_tokenizer.vit.vision_model.embeddings.position_embedding.weight"]
        size = int(round(math.sqrt(table.shape[0])))
        pe = table.reshape(size, size, -1).unsqueeze(0).permute(0, 3, 1, 2)
        pe = F.interpolate(pe, size=(grid_h, grid_w), mode="bicubic", align_corners=False)
        pe = pe.permute(0, 2, 3, 1).reshape(1, grid_h * grid_w, -1)
        pe = pe.reshape(
            1,
            grid_h // self.hidden_stride,
            self.hidden_stride,
            grid_w // self.hidden_stride,
            self.hidden_stride,
            -1,
        )
        pe = pe.permute(0, 1, 3, 2, 4, 5).reshape(grid_h * grid_w, -1)
        return pe

    def _rope(
        self,
        grid_thw: tuple[int, int, int],
        window_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build 2D rotary frequencies in window order."""
        _, grid_h, grid_w = grid_thw
        dim = self.hidden_size // self.heads // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        hpos = torch.arange(grid_h).unsqueeze(1).expand(-1, grid_w)
        hpos = hpos.reshape(grid_h // 2, 2, grid_w // 2, 2).permute(0, 2, 1, 3).flatten()
        wpos = torch.arange(grid_w).unsqueeze(0).expand(grid_h, -1)
        wpos = wpos.reshape(grid_h // 2, 2, grid_w // 2, 2).permute(0, 2, 1, 3).flatten()
        freqs = torch.cat([torch.outer(hpos.float(), inv_freq), torch.outer(wpos.float(), inv_freq)], dim=-1)
        freqs = freqs.reshape(-1, self.merge_unit, freqs.shape[-1])[window_index].reshape(-1, freqs.shape[-1])
        angles = torch.cat([freqs, freqs], dim=-1)
        return angles.cos(), angles.sin()

    def _window_plan(self, grid_thw: tuple[int, int, int]):
        """Return the window permutation and its sequence lengths."""
        grid_t, grid_h, grid_w = grid_thw
        units = self.window_size // self.hidden_stride // self.patch_size
        llm_h, llm_w = grid_h // self.hidden_stride, grid_w // self.hidden_stride
        index = torch.arange(grid_t * llm_h * llm_w).reshape(grid_t, llm_h, llm_w)
        pad_h = units - llm_h % units
        pad_w = units - llm_w % units
        num_h = (llm_h + pad_h) // units
        num_w = (llm_w + pad_w) // units
        padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        padded = padded.reshape(grid_t, num_h, units, num_w, units).permute(0, 1, 3, 2, 4)
        padded = padded.reshape(grid_t, num_h * num_w, units, units)
        seqlens = (padded != -100).sum([2, 3]).reshape(-1)
        flat = padded.reshape(-1)
        window_index = flat[flat != -100]
        cu_window = [0]
        cu_window.extend((seqlens.cumsum(0) * self.merge_unit + cu_window[0]).tolist())
        return window_index, cu_window

    @staticmethod
    def _mask(cu_seqlens: list[int]) -> torch.Tensor:
        """Block-diagonal additive mask for the given segments."""
        length = cu_seqlens[-1]
        mask = torch.full((length, length), -1e9)
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            mask[start:end, start:end] = 0.0
        return mask

    def vision(self, patches: np.ndarray, grid_thw: tuple[int, int, int]) -> torch.Tensor:
        """Run the vision tower on patch vectors."""
        w = self.w
        prefix = "visual_tokenizer.vit.vision_model"
        patch_w = w[f"{prefix}.embeddings.patch_embedding.weight"]
        patch_w = patch_w.reshape(patch_w.shape[0], -1) if patch_w.ndim == 4 else patch_w
        x = torch.from_numpy(patches).to(DTYPE) @ patch_w.T
        x = x + w[f"{prefix}.embeddings.patch_embedding.bias"]
        x = x + self.position_embedding(grid_thw[1], grid_thw[2])

        window_index, _cu_window = self._window_plan(grid_thw)
        n_patches = x.shape[0]
        x = x.reshape(n_patches // self.merge_unit, self.merge_unit, -1)[window_index].reshape(n_patches, -1)
        cos, sin = self._rope(grid_thw, window_index)
        mask = self._mask([0, grid_thw[0] * grid_thw[1] * grid_thw[2]])

        head_dim = self.hidden_size // self.heads
        for index in range(self.layers):
            base = f"{prefix}.encoder.layers.{index}"
            residual = x
            h = F.layer_norm(x, (self.hidden_size,), w[f"{base}.layer_norm1.weight"], w[f"{base}.layer_norm1.bias"], self.eps)
            q = (h @ w[f"{base}.self_attn.q_proj.weight"].T + w[f"{base}.self_attn.q_proj.bias"]).reshape(-1, self.heads, head_dim)
            k = (h @ w[f"{base}.self_attn.k_proj.weight"].T + w[f"{base}.self_attn.k_proj.bias"]).reshape(-1, self.heads, head_dim)
            v = (h @ w[f"{base}.self_attn.v_proj.weight"].T + w[f"{base}.self_attn.v_proj.bias"]).reshape(-1, self.heads, head_dim)

            def rotate(t):
                """Rotate the two halves of the last axis."""
                half = t.shape[-1] // 2
                return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

            q = q * cos[:, None, :] + rotate(q) * sin[:, None, :]
            k = k * cos[:, None, :] + rotate(k) * sin[:, None, :]

            qh = q.transpose(0, 1)
            kh = k.transpose(0, 1)
            vh = v.transpose(0, 1)
            attn = torch.softmax(qh @ kh.transpose(-1, -2) * head_dim**-0.5 + mask, dim=-1)
            out = (attn @ vh).transpose(0, 1).reshape(n_patches, self.hidden_size)
            x = residual + (out @ w[f"{base}.self_attn.out_proj.weight"].T + w[f"{base}.self_attn.out_proj.bias"])

            residual = x
            h = F.layer_norm(x, (self.hidden_size,), w[f"{base}.layer_norm2.weight"], w[f"{base}.layer_norm2.bias"], self.eps)
            h = F.gelu(h @ w[f"{base}.mlp.fc1.weight"].T + w[f"{base}.mlp.fc1.bias"], approximate="tanh")
            h = h @ w[f"{base}.mlp.fc2.weight"].T + w[f"{base}.mlp.fc2.bias"]
            x = residual + h

        reverse = torch.argsort(window_index)
        x = x.reshape(n_patches // self.merge_unit, self.merge_unit, -1)[reverse].reshape(n_patches, -1)
        return x

    def visual_tokens(self, patches: np.ndarray, grid_thw: tuple[int, int, int]) -> torch.Tensor:
        """Map an image to soft visual tokens."""
        features = self.vision(patches, grid_thw)
        features = features.reshape(features.shape[0] // self.merge_unit, -1)
        head = "visual_tokenizer.head"
        logits = features @ self.w[f"{head}.0.weight"].T
        logits = F.layer_norm(logits, (logits.shape[-1],), self.w[f"{head}.1.weight"], self.w[f"{head}.1.bias"], 1e-5)
        probs = torch.softmax(logits.float(), dim=-1).to(features.dtype)
        return torch.cat([probs, torch.zeros(probs.shape[0], 4, dtype=probs.dtype)], dim=-1)

    # ---------------- text ----------------
    def logits(self, ids: np.ndarray, visual: torch.Tensor | None) -> torch.Tensor:
        """Run the decoder over a prompt and return logits."""
        w = self.w
        ids_t = torch.from_numpy(ids.astype(np.int64))[None]
        embeds = w["llm.model.embed_tokens.weight"][torch.clamp(ids_t, min=0)]
        if visual is not None:
            embeds[ids_t == -300] = (visual @ w["vte.weight"]).to(embeds.dtype)
            for position, indicator in enumerate((-301, -302, -303, -304)):
                embeds[ids_t == indicator] = w["vte.weight"][65536 - 4 + position]
        hidden = embeds[0]
        heads, kv_heads, head_dim = 16, 8, 128
        seq = hidden.shape[0]
        causal = torch.full((seq, seq), float("-inf"))
        causal = torch.triu(causal, diagonal=1)
        for index in range(28):
            base = f"llm.model.layers.{index}"
            h = F.rms_norm(hidden, (1024,), w[f"{base}.input_layernorm.weight"], self.eps)
            q = (h @ w[f"{base}.self_attn.q_proj.weight"].T).reshape(seq, heads, head_dim)
            k = (h @ w[f"{base}.self_attn.k_proj.weight"].T).reshape(seq, kv_heads, head_dim)
            v = (h @ w[f"{base}.self_attn.v_proj.weight"].T).reshape(seq, kv_heads, head_dim)
            q = F.rms_norm(q, (head_dim,), w[f"{base}.self_attn.q_norm.weight"], self.eps)
            k = F.rms_norm(k, (head_dim,), w[f"{base}.self_attn.k_norm.weight"], self.eps)

            def rotate(t):
                """Rotate the two halves of the last axis."""
                half = t.shape[-1] // 2
                return torch.cat([-t[..., half:], t[..., :half]], dim=-1)

            inv = 1.0 / (1_000_000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
            pos = torch.arange(seq, dtype=torch.float32)
            ang = torch.outer(pos, inv)
            cos, sin = torch.cat([ang, ang], -1).cos(), torch.cat([ang, ang], -1).sin()
            q = q * cos[:, None, :] + rotate(q) * sin[:, None, :]
            k = k * cos[:, None, :] + rotate(k) * sin[:, None, :]

            kh = k.transpose(0, 1).repeat_interleave(heads // kv_heads, dim=0)
            vh = v.transpose(0, 1).repeat_interleave(heads // kv_heads, dim=0)
            attn = torch.softmax(q.transpose(0, 1) @ kh.transpose(-1, -2) * head_dim**-0.5 + causal, dim=-1)
            out = (attn @ vh).transpose(0, 1).reshape(seq, heads * head_dim)
            hidden = hidden + out @ w[f"{base}.self_attn.o_proj.weight"].T

            h = F.rms_norm(hidden, (1024,), w[f"{base}.post_attention_layernorm.weight"], self.eps)
            hidden = hidden + (
                F.silu(h @ w[f"{base}.mlp.gate_proj.weight"].T) * (h @ w[f"{base}.mlp.up_proj.weight"].T)
            ) @ w[f"{base}.mlp.down_proj.weight"].T

        hidden = F.rms_norm(hidden, (1024,), w["llm.model.norm.weight"], self.eps)
        return hidden @ w["llm.model.embed_tokens.weight"].T
