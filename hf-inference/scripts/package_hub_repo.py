#!/usr/bin/env python3
"""Build a Hub-ready VisionPsyNano folder from a local checkpoint.

  python scripts/package_hub_repo.py --source /path/to/ckpt --no-flash -o /tmp/VisionPsy-Nano-460M
  python scripts/package_hub_repo.py --source /path/to/ckpt --flash -o /tmp/VisionPsy-Nano-460M-Flash
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from safetensors.torch import save_model
import torch

from configuration_visionpsynano import VisionPsyNanoConfig
from hub_packaging import AUTO_MAP, copy_trust_remote_code
from modeling_visionpsynano import VisionPsyNanoForConditionalGeneration
from processing_visionpsynano import VisionPsyNanoProcessor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--source",
        required=True,
        help="Local checkpoint dir or Hub repo id (legacy or already-packaged)",
    )
    p.add_argument(
        "--flash",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Package as VisionPsyNano Flash (optimized preprocess). "
            "Use --no-flash for the default checkpoint. Default: infer from source."
        ),
    )
    p.add_argument("--output", "-o", required=True, help="Output directory for the Hub package")
    p.add_argument(
        "--push-to",
        default=None,
        help="Optional Hub repo id to upload the packaged folder to",
    )
    p.add_argument("--private", action="store_true", help="Create private Hub repo when pushing")
    p.add_argument("--revision", default=None, help="Hub revision for --source when remote")
    return p.parse_args()


def load_model(source: str, is_flash: bool | None, revision: str | None):
    config_path = None
    if os.path.isdir(source):
        config_path = os.path.join(source, "config.json")
    else:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(source, "config.json", revision=revision)

    with open(config_path) as f:
        raw = json.load(f)

    has_auto = isinstance(raw.get("auto_map"), dict) and raw.get("model_type") in ("visionpsy", "visionpsynano")
    if has_auto and is_flash is None:
        model = VisionPsyNanoForConditionalGeneration.from_pretrained(
            source, trust_remote_code=True
        )
        return model

    model = VisionPsyNanoForConditionalGeneration.from_legacy_pretrained(
        source, is_flash=is_flash, revision=revision
    )
    if is_flash is not None and model.config.is_flash != is_flash:
        new_cfg = VisionPsyNanoConfig.from_legacy_dict(raw, is_flash=is_flash)
        model.config = new_cfg
        model.cfg = new_cfg.to_vlm_config()
    return model


def main() -> None:
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    is_flash = args.flash

    print(f"[package] loading source={args.source} is_flash={is_flash}")
    model = load_model(args.source, is_flash, args.revision)
    config: VisionPsyNanoConfig = model.config
    if is_flash is not None:
        config = VisionPsyNanoConfig.from_legacy_dict(config.to_dict(), is_flash=is_flash)
        model.config = config
        model.cfg = config.to_vlm_config()

    config.architectures = ["VisionPsyNanoForConditionalGeneration"]
    config.auto_map = dict(AUTO_MAP)
    config.model_type = "visionpsynano"
    config.save_pretrained(args.output)

    cfg_path = os.path.join(args.output, "config.json")
    with open(cfg_path) as f:
        cfg_json = json.load(f)
    cfg_json["model_type"] = "visionpsynano"
    cfg_json["architectures"] = ["VisionPsyNanoForConditionalGeneration"]
    cfg_json["auto_map"] = dict(AUTO_MAP)
    cfg_json["is_flash"] = bool(config.is_flash)
    cfg_json["hf_repo_name"] = (
        "qvac/VisionPsy-Nano-460M-Flash" if config.is_flash else "qvac/VisionPsy-Nano-460M"
    )
    cfg_json["vlm_load_backbone_weights"] = False
    cfg_json["vlm_checkpoint_path"] = None
    cfg_json.pop("variant", None)
    cfg_json.pop("text_config", None)
    cfg_json.pop("vision_config", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg_json, f, indent=2)
        f.write("\n")

    import shutil

    license_src = os.path.join(os.path.dirname(_ROOT), "LICENSE.txt")
    if not os.path.isfile(license_src):
        license_src = os.path.join(_ROOT, "LICENSE.txt")
    if os.path.isfile(license_src):
        shutil.copy2(license_src, os.path.join(args.output, "LICENSE"))

    weights_path = os.path.join(args.output, "model.safetensors")
    if getattr(model.cfg, "lm_tie_weights", True):
        with torch.no_grad():
            shared = model.decoder.head.weight.detach().clone()
            model.decoder.token_embedding.weight = torch.nn.Parameter(shared.clone())
            model.decoder.head.weight = torch.nn.Parameter(shared.clone())
    save_model(model, weights_path)
    if getattr(model.cfg, "lm_tie_weights", True):
        model.tie_weights()
    print(f"[package] wrote {weights_path}")

    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ):
        if os.path.isdir(args.source):
            src_f = os.path.join(args.source, name)
            if os.path.exists(src_f):
                shutil.copy2(src_f, os.path.join(args.output, name))

    processor = VisionPsyNanoProcessor.from_config(config)
    if not os.path.exists(os.path.join(args.output, "tokenizer.json")):
        processor.tokenizer.save_pretrained(args.output)
    else:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.output)
        VisionPsyNanoProcessor._attach_extra_token_attrs(tok, config.vlm_extra_tokens)
        processor.tokenizer = tok

    processor.save_pretrained(args.output)
    proc_cfg_path = os.path.join(args.output, "processor_config.json")
    proc_cfg = {}
    if os.path.exists(proc_cfg_path):
        with open(proc_cfg_path) as f:
            proc_cfg = json.load(f)
    proc_cfg["processor_class"] = "VisionPsyNanoProcessor"
    proc_cfg["auto_map"] = {
        "AutoProcessor": "processing_visionpsynano.VisionPsyNanoProcessor"
    }
    with open(proc_cfg_path, "w") as f:
        json.dump(proc_cfg, f, indent=2)
        f.write("\n")
    print(f"[package] is_flash={config.is_flash}")

    copied = copy_trust_remote_code(args.output)
    print(f"[package] copied trust_remote_code assets: {copied}")

    attributions_path = os.path.join(args.output, "ATTRIBUTIONS.md")
    if not os.path.exists(attributions_path):
        with open(attributions_path, "w") as f:
            f.write(
                "# Attributions\n\n"
                "VisionPsyNano builds on publicly available components:\n\n"
                "- **Language model backbone:** "
                "[HuggingFaceTB/SmolLM2-360M-Instruct]"
                "(https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)\n"
                "- **Vision encoder backbone:** "
                "[google/siglip2-base-patch16-512]"
                "(https://huggingface.co/google/siglip2-base-patch16-512)\n\n"
                "Please refer to those projects for their respective licenses and notices.\n"
            )

    readme = os.path.join(args.output, "README.md")
    if not os.path.exists(readme):
        title = "VisionPsyNano Flash" if config.is_flash else "VisionPsyNano"
        hub_id = args.push_to or (
            "qvac/VisionPsy-Nano-460M-Flash" if config.is_flash else "qvac/VisionPsy-Nano-460M"
        )
        sibling = (
            "qvac/VisionPsy-Nano-460M"
            if config.is_flash
            else "qvac/VisionPsy-Nano-460M-Flash"
        )
        if config.is_flash:
            flash_note = (
                "This checkpoint enables **Flash** optimized image preprocessing "
                f"(`is_flash=true`). For the default preprocess, use "
                f"[`{sibling}`](https://huggingface.co/{sibling})."
            )
        else:
            flash_note = (
                "This is the default VisionPsyNano checkpoint (`is_flash=false`). "
                f"For Flash optimized preprocessing, use "
                f"[`{sibling}`](https://huggingface.co/{sibling})."
            )
        tags = [
            "vision-language",
            "multimodal",
            "visionpsy",
            "visionpsynano",
        ]
        if config.is_flash:
            tags.append("flash")
        tags_yaml = "\n".join(f"  - {t}" for t in tags)
        with open(readme, "w") as f:
            f.write(
                f"""---
library_name: transformers
pipeline_tag: image-text-to-text
license: apache-2.0
tags:
{tags_yaml}
---

# {title}

{flash_note}

Same **VisionPsyNano** architecture and class names (`VisionPsyNano*`); Flash only changes
image preprocessing, not the model class.

Built on [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)
and [SigLIP2](https://huggingface.co/google/siglip2-base-patch16-512). See `ATTRIBUTIONS.md`.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

repo = "{hub_id}"
model = AutoModelForImageTextToText.from_pretrained(
    repo, trust_remote_code=True, dtype="auto"
).cuda().eval()
model.apply_deploy_profile(model.device)
processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)

inputs = processor(images=Image.open("image.jpg"), text="Describe the image.", return_tensors="pt")
inputs = {{k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items() if v is not None}}
inputs.pop("pixel_values", None)
out = model.generate(**inputs, max_new_tokens=128, greedy=True)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

## License

Apache License 2.0. See `LICENSE`.
"""
            )

    print(f"[package] done → {args.output}")

    if args.push_to:
        from huggingface_hub import create_repo, upload_folder

        url = create_repo(args.push_to, private=args.private, exist_ok=True)
        print(f"[package] uploading to {url.repo_id}")
        upload_folder(
            repo_id=url.repo_id,
            folder_path=args.output,
            repo_type="model",
            commit_message=(
                f"Add Transformers trust_remote_code packaging "
                f"(VisionPsyNano{' Flash' if config.is_flash else ''})"
            ),
        )
        print(f"[package] pushed https://huggingface.co/{url.repo_id}")


if __name__ == "__main__":
    main()
