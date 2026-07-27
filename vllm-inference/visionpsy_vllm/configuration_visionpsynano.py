"""HF-style config for the vLLM VisionPsy port.

vLLM resolves a model class from `config.architectures` and reads
`config.text_config` for the reused Llama backbone. VisionPsy checkpoints ship
a flat config (vit_*, lm_*, ...); this class normalizes it at load time:
  * flat lm_* fields -> a standard nested LlamaConfig (`text_config`)
  * flat vit_* fields -> consumed by the ported ViT
  * image_token_id -> resolved lazily from the checkpoint's own tokenizer.json
No vLLM dependency here (pure transformers).
"""
import json
import os.path as osp

from transformers import PretrainedConfig
from transformers.models.llama.configuration_llama import LlamaConfig


def _added_token_id(name_or_path, token):
    """Exact id of an added special token, read from the checkpoint's tokenizer.json."""
    if osp.isdir(name_or_path):
        path = osp.join(name_or_path, "tokenizer.json")
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=name_or_path, filename="tokenizer.json")
    with open(path) as f:
        data = json.load(f)
    for t in data.get("added_tokens", []):
        if t.get("content") == token:
            return int(t["id"])
    vocab_id = data.get("model", {}).get("vocab", {}).get(token)
    if vocab_id is None:
        raise ValueError(f"token {token!r} not found in {path}")
    return int(vocab_id)


class VisionPsyNanoConfig(PretrainedConfig):
    model_type = "visionpsynano"
    # so AutoConfig recurses into the nested text config
    sub_configs = {"text_config": LlamaConfig}

    def __init__(
        self,
        text_config=None,
        # vision tower (SigLIP-style) hyper-params for the ported ViT
        vit_hidden_dim=768,
        vit_inter_dim=3072,
        vit_patch_size=16,
        vit_img_size=512,
        vit_n_heads=12,
        vit_n_blocks=12,
        vit_dropout=0.0,
        vit_ln_eps=1e-6,
        vit_cls_flag=False,
        # modality projector
        mp_pixel_shuffle_factor=4,
        mp_image_token_length=64,
        # multimodal glue
        image_token_id=None,
        **kwargs,
    ):
        if text_config is None and "lm_hidden_dim" in kwargs:
            # Checkpoints ship flat lm_* fields; build the nested Llama config.
            head_dim = kwargs["lm_hidden_dim"] // kwargs["lm_n_heads"]
            rope_theta = kwargs.get("lm_re_base", 100000)
            text_config = LlamaConfig(
                hidden_size=kwargs["lm_hidden_dim"],
                intermediate_size=kwargs["lm_inter_dim"],
                num_hidden_layers=kwargs["lm_n_blocks"],
                num_attention_heads=kwargs["lm_n_heads"],
                num_key_value_heads=kwargs["lm_n_kv_heads"],
                head_dim=head_dim,
                vocab_size=kwargs["lm_vocab_size"],
                max_position_embeddings=kwargs["lm_max_position_embeddings"],
                rms_norm_eps=kwargs["lm_rms_eps"],
                # transformers 5.x routes RoPE through `rope_parameters` (the bare
                # `rope_theta` kwarg is ignored -> serialized as null). Set both so
                # whichever vLLM/transformers reads sees the trained theta.
                rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
                # The checkpoint stores BOTH decoder.token_embedding.weight and
                # decoder.head.weight (identical, tied at train time). With
                # tie_word_embeddings=True, vLLM's Llama loader SKIPS loading both
                # embed_tokens and lm_head (leaving them zero -> all-zero logits).
                # Declaring them untied makes vLLM load both from the checkpoint;
                # since the saved tensors are identical, behaviour is unchanged.
                tie_word_embeddings=False,
                hidden_act="silu",
                attention_bias=False,
                mlp_bias=False,
            )
            text_config.rope_theta = rope_theta
        if text_config is None:
            text_config = {}
        if isinstance(text_config, dict):
            text_config = LlamaConfig(**text_config)
        self.text_config = text_config

        self.vit_hidden_dim = vit_hidden_dim
        self.vit_inter_dim = vit_inter_dim
        self.vit_patch_size = vit_patch_size
        self.vit_img_size = vit_img_size
        self.vit_n_heads = vit_n_heads
        self.vit_n_blocks = vit_n_blocks
        self.vit_dropout = vit_dropout
        self.vit_ln_eps = vit_ln_eps
        self.vit_cls_flag = vit_cls_flag

        self.mp_pixel_shuffle_factor = mp_pixel_shuffle_factor
        self.mp_image_token_length = mp_image_token_length

        self.image_token_id = image_token_id

        super().__init__(**kwargs)

    def resolved_image_token_id(self):
        # The checkpoint does not store image_token_id; its own tokenizer.json
        # is the source of truth (vocab order differs from the vlm_extra_tokens
        # dict order). Resolved lazily because _name_or_path is only set after
        # from_pretrained().
        if self.image_token_id is None:
            image_token = (getattr(self, "vlm_extra_tokens", None) or {}).get(
                "image_token", "<|image|>"
            )
            self.image_token_id = _added_token_id(self._name_or_path, image_token)
        return self.image_token_id
