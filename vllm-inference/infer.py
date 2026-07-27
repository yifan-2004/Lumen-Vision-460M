#!/usr/bin/env python
"""VisionPsyNano - one-shot image+prompt inference via vLLM (offline engine).

Mirrors the reference preprocessing exactly: the image is dynamically resized and
split into 512x512 tiles client-side, the prompt carries the global/tile position
tokens with one image placeholder per tile, and the plugin expands each placeholder
to mp_image_token_length image tokens inside vLLM.

    python infer.py --ckpt qvac/VisionPsy-Nano-460M \
        --image img.jpg --prompt "Please describe the image."
"""
import argparse
import json
import os.path as osp
import sys
from dataclasses import fields

ROOT = osp.dirname(osp.abspath(__file__))
for _p in (ROOT, osp.join(ROOT, "visionpsy_vllm", "reference")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="qvac/VisionPsy-Nano-460M",
                    help="HF repo id or local folder of a Hub-packaged VisionPsyNano checkpoint")
    ap.add_argument("--image", default=osp.join(ROOT, "img.jpg"))
    ap.add_argument("--prompt", default="Please describe the image.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    from PIL import Image

    import visionpsy_vllm
    visionpsy_vllm.register()

    from vllm import LLM, SamplingParams

    # Build the engine BEFORE the tokenizer / image preprocessing: vLLM forks its
    # engine subprocess, and forking after HF-tokenizers / torch have spawned
    # threads deadlocks the child.
    llm = LLM(
        model=args.ckpt,
        dtype="float32",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 128},
        enforce_eager=True,
    )

    from models.config import VLMConfig
    from data.processors import (
        apply_model_preprocess,
        get_image_processor,
        get_image_string,
        get_tokenizer,
    )

    if osp.isdir(args.ckpt):
        cfg_path = osp.join(args.ckpt, "config.json")
    else:
        from huggingface_hub import hf_hub_download
        cfg_path = hf_hub_download(repo_id=args.ckpt, filename="config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    valid = {fld.name for fld in fields(VLMConfig)}
    cfg = VLMConfig(**{k: v for k, v in raw.items() if k in valid})

    tokenizer = get_tokenizer(
        cfg.lm_tokenizer, getattr(cfg, "vlm_extra_tokens", None),
        getattr(cfg, "lm_chat_template", None),
    )
    # Older checkpoints were trained without image splitting and lack these keys.
    if "vlm_extra_tokens" in raw:
        apply_model_preprocess(cfg)  # resolves the base/Flash resize policy
        max_img = cfg.inference_max_img_size or cfg.max_img_size
        resize_to_max = cfg.resize_to_max_side_len
        min_side = cfg.resize_min_side_len
    else:
        max_img = cfg.vit_img_size
        resize_to_max = False
        min_side = None
    image_processor = get_image_processor(
        max_img, cfg.vit_img_size, resize_to_max, min_side,
    )

    img = Image.open(args.image).convert("RGB")
    tiles, ratio = image_processor(img)
    if (not hasattr(tokenizer, "global_image_token")
            and ratio[0] * ratio[1] == len(tiles) - 1):
        tiles = tiles[1:]
    # One placeholder per tile; the plugin expands each to mp_image_token_length.
    image_string = get_image_string(tokenizer, [ratio], 1)
    pil_tiles = [
        Image.fromarray(
            (t.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
        )
        for t in tiles
    ]

    messages = [{"role": "user", "content": image_string + args.prompt}]
    prompt = tokenizer.apply_chat_template(
        [messages], tokenize=False, add_generation_prompt=True,
    )
    if isinstance(prompt, list):
        prompt = prompt[0]

    outputs = llm.generate(
        {"prompt": prompt, "multi_modal_data": {"image": pil_tiles}},
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    text = outputs[0].outputs[0].text.strip()

    print()
    print("---- model output ----")
    print(text)
    print("----------------------")


if __name__ == "__main__":
    main()
