"""vLLM model class for VisionPsy (verified against vllm==0.22.0 source).

  * vision_encoder (ViT) + MP (ModalityProjector): ported verbatim, run as
    plain modules -> exact parity; checkpoint names `vision_encoder.*` / `MP.*` load 1:1.
  * language_model: vLLM's registered Llama (SmolLM2 == Llama) -> inherits paged
    attention / KV cache / continuous batching. `decoder.*` ckpt names are remapped to
    the Llama layout and handed to the LM's own loader (fuses qkv / gate_up).

Interface matches the in-tree `llava.py` pattern: implement `embed_multimodal`, inherit
`embed_input_ids`/merge from `SupportsMultiModal`, delegate `forward` to
`language_model.model(...)`.
"""
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import SupportsMultiModal, SupportsPP
from vllm.model_executor.models.utils import init_vllm_registered_model, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY

from .vision import ViT, ModalityProjector
from .processing_visionpsynano import (
    VisionPsyNanoDummyInputsBuilder,
    VisionPsyNanoMultiModalProcessor,
    VisionPsyNanoProcessingInfo,
)


@MULTIMODAL_REGISTRY.register_processor(
    VisionPsyNanoMultiModalProcessor,
    info=VisionPsyNanoProcessingInfo,
    dummy_inputs=VisionPsyNanoDummyInputsBuilder,
)
class VisionPsyNanoForConditionalGeneration(nn.Module, SupportsMultiModal, SupportsPP):
    # Tell AutoWeightsLoader / the LM loader how separate q/k/v and gate/up fuse.
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int):
        if modality.startswith("image"):
            return "<|image|>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        if hasattr(config, "resolved_image_token_id"):
            self.image_token_id = config.resolved_image_token_id()
        else:
            self.image_token_id = config.image_token_id

        self.configure_mm_token_handling(
            vocab_size=config.text_config.vocab_size,
            mm_token_ids=[self.image_token_id],
        )

        with self._mark_tower_model(vllm_config, "image"):
            self.vision_encoder = ViT(
                vit_hidden_dim=config.vit_hidden_dim,
                vit_inter_dim=config.vit_inter_dim,
                vit_patch_size=config.vit_patch_size,
                vit_img_size=config.vit_img_size,
                vit_n_heads=config.vit_n_heads,
                vit_n_blocks=config.vit_n_blocks,
                vit_dropout=config.vit_dropout,
                vit_ln_eps=config.vit_ln_eps,
                vit_cls_flag=config.vit_cls_flag,
            )
            self.MP = ModalityProjector(
                vit_hidden_dim=config.vit_hidden_dim,
                lm_hidden_dim=config.text_config.hidden_size,
                pixel_shuffle_factor=config.mp_pixel_shuffle_factor,
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                architectures=["LlamaForCausalLM"],
                prefix=maybe_prefix(prefix, "language_model"),
            )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    # ---- multimodal: each image item is one 512x512 tile -> 64 feature tokens ----
    def embed_multimodal(self, **kwargs):
        pixel_values = kwargs.get("pixel_values")
        if pixel_values is None:
            return []
        if isinstance(pixel_values, (list, tuple)):
            pixel_values = torch.cat([p.to(self._vit_dtype()) for p in pixel_values], dim=0)
        else:
            pixel_values = pixel_values.to(self._vit_dtype())
        feats = self.vision_encoder(pixel_values)   # [N, 1024, vit_hidden]
        img_embeds = self.MP(feats)                 # [N, 64, lm_hidden]
        return img_embeds                           # 3D -> flattened to N*64 by vLLM

    def _vit_dtype(self):
        return self.vision_encoder.patch_embedding.conv.weight.dtype

    def get_language_model(self):
        return self.language_model

    # ---- runner contract (mirror llava.py) ----
    def forward(self, input_ids, positions, intermediate_tensors=None,
                inputs_embeds=None, **kwargs):
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.language_model.model(
            input_ids, positions, intermediate_tensors, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.language_model.compute_logits(hidden_states)

    # ---- weight loading: vision_encoder/MP load 1:1; decoder.* -> Llama ----
    @staticmethod
    def _remap_decoder_name(name: str) -> str:
        n = name[len("decoder."):]
        if n == "token_embedding.weight":
            return "model.embed_tokens.weight"
        if n == "norm.weight":
            return "model.norm.weight"
        if n == "head.weight":
            return "lm_head.weight"
        if n.startswith("blocks."):
            idx, sub = n[len("blocks."):].split(".", 1)
            sub = (sub
                   .replace("attn.q_proj", "self_attn.q_proj")
                   .replace("attn.k_proj", "self_attn.k_proj")
                   .replace("attn.v_proj", "self_attn.v_proj")
                   .replace("attn.out_proj", "self_attn.o_proj")
                   .replace("norm1", "input_layernorm")
                   .replace("norm2", "post_attention_layernorm"))
            return f"model.layers.{idx}.{sub}"
        return n

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        lm_weights = []
        embed_src = None  # tied token-embedding matrix (token_embedding == head)
        for name, w in weights:
            if name.startswith("vision_encoder.") or name.startswith("MP."):
                if name in params:
                    p = params[name]
                    getattr(p, "weight_loader", default_weight_loader)(p, w)
                    loaded.add(name)
            elif name.startswith("decoder."):
                # vLLM's Llama computes RoPE internally; the checkpoint's saved
                # rotary buffer (decoder.rotary_embd.inv_freq) has no home there.
                if name.startswith("decoder.rotary_embd"):
                    continue
                rn = self._remap_decoder_name(name)
                # token_embedding and head are tied (identical); safetensors may
                # dedup one out of the stream. Capture whichever is yielded.
                if rn in ("model.embed_tokens.weight", "lm_head.weight"):
                    embed_src = w
                lm_weights.append((rn, w))
        lm_loaded = set(self.language_model.load_weights(lm_weights) or ())
        loaded.update(f"language_model.{k}" for k in lm_loaded)

        # vLLM's AutoWeightsLoader skips model.embed_tokens.weight for this config
        # (it leaves the VocabParallelEmbedding zero-initialised -> all-zero text
        # embeddings). Load it directly from the checkpoint tensor.
        # embed_tokens' source (decoder.token_embedding.weight) is deduped out of
        # the stream by safetensors (tied with decoder.head.weight). Load it from
        # the captured tied matrix so text-token embeddings aren't zero.
        if "model.embed_tokens.weight" not in lm_loaded and embed_src is not None:
            ep = dict(self.language_model.named_parameters()).get("model.embed_tokens.weight")
            if ep is not None:
                getattr(ep, "weight_loader", default_weight_loader)(ep, embed_src)
                loaded.add("language_model.model.embed_tokens.weight")
        return loaded
