#!/usr/bin/env python3
"""Lumen: local single-image or folder inference with JSONL and HTML reports."""
import argparse
import base64
import html
import importlib.util
import io
import json
import os
from pathlib import Path
import sys

DEFAULT_MODEL = 'yifanouyang/Lumen-Vision-460M'
DEFAULT_REVISION = 'd8e4540a883a5ce45cd26d36e42d4cad58ab5aa4'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def collect_images(image=None, folder=None):
    if image:
        path = Path(image).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Image not found: {path}')
        return [path]
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f'Folder not found: {path}')
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError(f'No supported images in {path}')
    return files


def choose_device(torch, requested):
    if requested == 'auto':
        # Reference runtime is tested on CPU; select CUDA when available.
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; select --device cpu')
    if requested == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS is unavailable; select --device cpu')
    return requested


def render_report(rows, target):
    cards = []
    for row in rows:
        preview = ''
        if row.get('thumbnail'):
            preview = f'<img src="data:image/jpeg;base64,{row["thumbnail"]}" alt="Input preview">'
        cards.append('<article>' + preview + '<h2>' + html.escape(row['image']) + '</h2><pre>'
                     + html.escape(row.get('answer', row.get('error', ''))) + '</pre></article>')
    Path(target).write_text('<!doctype html><meta charset="utf-8"><title>Lumen report</title>'
        '<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#f4f6fa}'
        'article{background:white;padding:20px;margin:16px 0;border-radius:12px}'
        'img{max-width:100%;max-height:360px}pre{white-space:pre-wrap}h2{font-size:16px}</style>'
        '<h1>Lumen image report</h1>' + ''.join(cards), encoding='utf-8')


def load_runtime(args):
    if args.offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from huggingface_hub import snapshot_download
    runtime = Path(__file__).resolve().parents[1] / 'hf-inference'
    sys.path.insert(0, str(runtime))
    from configuration_visionpsynano import VisionPsyNanoConfig
    from modeling_visionpsynano import VisionPsyNanoForConditionalGeneration
    from processing_visionpsynano import VisionPsyNanoProcessor
    device = choose_device(torch, args.device)
    checkpoint = Path(args.model).expanduser()
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download(args.model, revision=args.revision,
            local_files_only=args.offline, allow_patterns=['*.json', '*.safetensors', '*.jinja']))
    # Constructor previously fetched a separate base tokenizer even when all
    # checkpoint files were local. Reuse the bundled tokenizer from the start.
    config = VisionPsyNanoConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.lm_tokenizer = str(checkpoint)
    model = VisionPsyNanoForConditionalGeneration.from_pretrained(str(checkpoint),
        config=config, local_files_only=True, dtype=torch.float32).to(device).eval()
    model.apply_eager_profile()
    processor = VisionPsyNanoProcessor.from_pretrained(str(checkpoint), local_files_only=True)
    model.set_tokenizer(processor.tokenizer)
    if getattr(args, 'projector', None):
        from safetensors.torch import load_file
        model.MP.load_state_dict(load_file(args.projector, device=device), strict=True)
    return torch, model, processor, device


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--image')
    source.add_argument('--folder', help='Process supported files in one folder, non-recursively')
    parser.add_argument('--prompt', default='Describe this image.')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--revision', default=DEFAULT_REVISION, help='Use main explicitly for another Hub model')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--projector', help='Optional locally trained projector.safetensors')
    parser.add_argument('--doctor', action='store_true', help='Check dependencies without downloading a model')
    parser.add_argument('--output', help='JSONL path; overwrites an existing file')
    parser.add_argument('--html', help='Self-contained report path')
    args = parser.parse_args(argv)
    if args.doctor:
        packages = ['torch', 'torchvision', 'transformers', 'huggingface_hub', 'PIL', 'safetensors', 'einops', 'numpy']
        checks = {p: importlib.util.find_spec(p) is not None for p in packages}
        print(json.dumps({'python': sys.version.split()[0], 'dependencies': checks}, indent=2))
        return 0 if all(checks.values()) and sys.version_info >= (3, 10) else 1
    if not (args.image or args.folder):
        parser.error('provide --image or --folder, or use --doctor')
    if not args.prompt.strip() or not 1 <= args.max_new_tokens <= 512:
        parser.error('provide a nonempty prompt and --max-new-tokens between 1 and 512')
    try:
        images = collect_images(args.image, args.folder)
    except ValueError as exc:
        parser.error(str(exc))
    targets = [Path(name).expanduser().resolve() for name in (args.output, args.html) if name]
    if len(set(targets)) != len(targets) or any(path in images for path in targets):
        parser.error('output paths must be distinct and must not overwrite an input image')
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageOps
    torch, model, processor, device = load_runtime(args)
    rows = []
    stream = open(args.output, 'w', encoding='utf-8') if args.output else None
    try:
        for path in images:
            row = {'image': str(path), 'prompt': args.prompt, 'model': args.model,
                   'revision': None if Path(args.model).is_dir() else args.revision, 'max_new_tokens': args.max_new_tokens,
                   'projector': args.projector}
            try:
                with Image.open(path) as original:
                    image = ImageOps.exif_transpose(original).convert('RGB')
                inputs = processor(images=image, text=args.prompt, return_tensors='pt', truncation=False)
                if inputs['input_ids'].shape[1] + args.max_new_tokens > model.config.lm_max_position_embeddings:
                    raise ValueError('Prompt plus generation exceeds model context; shorten the prompt.')
                inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items() if v is not None}
                inputs.pop('pixel_values', None)
                with torch.inference_mode():
                    generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, greedy=True)
                row['answer'] = processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
                if args.html:
                    image.thumbnail((512, 512)); buffer = io.BytesIO(); image.save(buffer, format='JPEG')
                    row['thumbnail'] = base64.b64encode(buffer.getvalue()).decode('ascii')
            except (OSError, ValueError) as exc:
                row['error'] = str(exc)
            if stream:
                stream.write(json.dumps({k:v for k,v in row.items() if k != 'thumbnail'}, ensure_ascii=False) + '\n')
                stream.flush()
            print(json.dumps({k:v for k,v in row.items() if k != 'thumbnail'}, ensure_ascii=False))
            rows.append(row)
    finally:
        if stream:
            stream.close()
    if args.html:
        render_report(rows, args.html)
    return 1 if any('error' in row for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
