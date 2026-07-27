"""vLLM multimodal processor for VisionPsy (vllm==0.22.0).

Image tile-splitting happens client-side (each image item handed to vLLM is already
one 512x512 tile, and the prompt already carries the global/tile position tokens with
ONE `<|image|>` placeholder per tile). So this processor:
  * fully overrides `_call_hf_processor` -> tokenize the prompt + ToTensor([0,1]) each tile,
  * expands each `<|image|>` placeholder to `mp_image_token_length` (64) image tokens,
  * declares pixel_values as a per-image batched field.
"""
from collections.abc import Mapping, Sequence

import numpy as np
import torch
from transformers import BatchFeature

from vllm.inputs import MultiModalDataDict
from vllm.multimodal.inputs import MultiModalFieldConfig
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    PromptReplacement,
    PromptUpdate,
)


class _StubProcessor:
    """Minimal stand-in for an HF processor. Only carries the
    tokenizer; image handling is done in `_call_hf_processor`."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class VisionPsyNanoProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}

    def get_hf_processor(self, **kwargs):
        return _StubProcessor(self.get_tokenizer())


class VisionPsyNanoDummyInputsBuilder(BaseDummyInputsBuilder[VisionPsyNanoProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return "<|image|>" * mm_counts.get("image", 0)

    def get_dummy_mm_data(self, seq_len, mm_counts, mm_options=None) -> MultiModalDataDict:
        size = self.info.get_hf_config().vit_img_size
        return {
            "image": self._get_dummy_images(
                width=size, height=size, num_images=mm_counts.get("image", 0),
            )
        }


class VisionPsyNanoMultiModalProcessor(BaseMultiModalProcessor[VisionPsyNanoProcessingInfo]):
    def _call_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs) -> BatchFeature:
        tokenizer = self.info.get_tokenizer()
        ids = tokenizer(prompt).input_ids  # match reference tokenizer() defaults
        out = {"input_ids": [ids]}
        # vLLM passes image data to the HF processor under the plural key "images".
        images = mm_data.get("images") or mm_data.get("image")
        if images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            # ToTensor-equivalent: HWC uint8 -> CHW float in [0,1] (no normalization,
            # matching the reference preprocessing's transforms.ToTensor()).
            tiles = []
            for img in images:
                arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
                tiles.append(torch.from_numpy(arr).permute(2, 0, 1).contiguous())
            out["pixel_values"] = tiles  # list of [3, H, W] (Pixtral-style per-item)
        return BatchFeature(out)

    def _hf_processor_applies_updates(self, prompt_text, mm_items,
                                      hf_processor_mm_kwargs, tokenization_kwargs) -> bool:
        # We tokenize with ONE placeholder per tile; vLLM must expand to 64.
        return False

    def _get_mm_fields_config(self, hf_inputs, hf_processor_mm_kwargs):
        return dict(pixel_values=MultiModalFieldConfig.batched("image"))

    def _get_prompt_updates(self, mm_items, hf_processor_mm_kwargs, out_mm_kwargs) -> Sequence[PromptUpdate]:
        hf_config = self.info.get_hf_config()
        if hasattr(hf_config, "resolved_image_token_id"):
            image_token_id = hf_config.resolved_image_token_id()
        else:
            image_token_id = hf_config.image_token_id
        n = hf_config.mp_image_token_length
        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=[image_token_id] * n,
            )
        ]
