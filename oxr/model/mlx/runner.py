"""OXR-1.0 runtime in MLX: tokenize -> visual tokens -> greedy decode."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image
from transformers import AutoTokenizer

from .model import EOS_IDS, IMAGE_PLACEHOLDER_ID, INDICATOR_IDS, OXR, VISUAL_ATOM_ID
from .preprocess import preprocess

DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}

IMAGE_TOKEN = "<image>"
TEXT_RECOGNITION = "Recognize the content of the image."
WHOLE_PAGE = "Recognize the whole page content of the image."


def render_chat(prompt: str, with_image: bool = True) -> str:
    """Equivalent of the repo's chat_template.jinja for a single user turn."""
    content = f"{IMAGE_TOKEN}{prompt}" if with_image else prompt
    return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def finish_reason_for(next_id: int) -> str:
    """EOS ends generation cleanly, even when it arrives at the token cap."""
    return "stop" if next_id in EOS_IDS else "length"


@dataclass(frozen=True)
class Generation:
    text: str
    finish_reason: str  # "stop" (EOS) or "length" (token budget exhausted)


class OXRRunner:
    def __init__(self, model_dir: Path | str, dtype: str = "bfloat16"):
        """Load the checkpoint and tokenizer into MLX."""
        if dtype not in DTYPES:
            raise ValueError(f"unsupported dtype {dtype!r}; expected one of {', '.join(DTYPES)}")
        self.model_dir = Path(model_dir)
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), trust_remote_code=False)
        raw = mx.load(str(self.model_dir / "model.safetensors"))
        weights = {name: array.astype(DTYPES[dtype]) for name, array in raw.items()}
        self.model = OXR(weights)

    # ---- input assembly -------------------------------------------------
    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text, marking image placeholders."""
        chunks = [self.tokenizer(chunk, add_special_tokens=False).input_ids for chunk in text.split(IMAGE_TOKEN)]
        ids = list(chunks[0])
        for chunk in chunks[1:]:
            ids.append(IMAGE_PLACEHOLDER_ID)
            ids.extend(chunk)
        return ids

    def encode(self, image: PIL.Image.Image, prompt: str) -> tuple[np.ndarray, mx.array]:
        """Build the prompt ids and the image's visual embeddings."""
        patches, grid_thw = preprocess(image)
        atoms = (grid_thw[0] * grid_thw[1] * grid_thw[2]) // 4
        ids = self._tokenize(render_chat(prompt))

        merged: list[int] = []
        for token in ids:
            if token == IMAGE_PLACEHOLDER_ID:
                merged.append(INDICATOR_IDS[0])
                merged.extend([VISUAL_ATOM_ID] * atoms)
                merged.append(INDICATOR_IDS[1])
            else:
                merged.append(token)
        visual = self.model.image_embeds(patches, grid_thw)
        return np.array(merged, dtype=np.int64), visual

    # ---- generation -----------------------------------------------------
    def generate(
        self,
        image: PIL.Image.Image,
        prompt: str = TEXT_RECOGNITION,
        max_tokens: int = 4096,
        repetition_penalty: float = 1.0,
    ) -> Generation:
        """Greedy decode one prompt and return text plus finish reason."""
        ids, visual = self.encode(image, prompt)
        mx.eval(visual)

        cache: list = [None] * self.model.llm.config.num_layers
        embeds = self.model.merge(ids, visual)
        logits = self.model.llm(embeds, cache, 0)
        produced: list[int] = []
        next_id = _pick(logits[-1], produced, repetition_penalty)

        length = len(ids)
        while next_id not in EOS_IDS and len(produced) < max_tokens:
            produced.append(next_id)
            token_embed = self.model.llm.embed[mx.array([next_id])]
            logits = self.model.llm(token_embed, cache, length)
            next_id = _pick(logits[-1], produced, repetition_penalty)
            length += 1

        return Generation(
            text=self.tokenizer.decode(produced, skip_special_tokens=True),
            finish_reason=finish_reason_for(next_id),
        )


def _pick(logits: mx.array, produced: list[int], penalty: float) -> int:
    """Greedy pick with an optional HF-style repetition penalty."""
    if penalty == 1.0 or not produced:
        return int(mx.argmax(logits).item())
    scores = logits.astype(mx.float32)
    seen = mx.array(sorted(set(produced)))
    values = scores[seen]
    penalised = mx.where(values > 0, values / penalty, values * penalty)
    scores[seen] = penalised
    return int(mx.argmax(scores).item())

