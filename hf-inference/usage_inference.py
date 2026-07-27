#!/usr/bin/env python3
"""VisionPsyNano — Transformers inference."""
from __future__ import annotations

import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

DEFAULT_MODEL = os.environ.get("VISIONPSY_NANO", "qvac/VisionPsy-Nano-460M")
_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_IMAGE = os.path.join(_ROOT, "data", "img.jpg")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--prompt", default="Please describe the image.")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--device", default=None)
    p.add_argument("--deploy", action="store_true", help="torch.compile + CUDA graphs")
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype="auto" if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    if args.deploy:
        model.apply_deploy_profile(device)
    else:
        model.apply_eager_profile()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, text=args.prompt, return_tensors="pt")
    inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in inputs.items()
        if v is not None
    }
    inputs.pop("pixel_values", None)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, greedy=True)
    print(processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip())


if __name__ == "__main__":
    main()
