#!/usr/bin/env python3
"""Lumen connector adaptation: frozen vision/decoder, answer-only shifted loss."""
import argparse
import json
import hashlib
import math
from pathlib import Path
import random
from types import SimpleNamespace


def read_data(path):
    path = Path(path).resolve()
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ('id', 'image', 'question', 'answer'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'{path}:{line_number}: nonempty {key} required')
        if row['id'] in seen:
            raise ValueError('duplicate id: ' + row['id'])
        seen.add(row['id'])
        row['image'] = str((path.parent / row['image']).resolve())
        if not Path(row['image']).is_file():
            raise ValueError('image missing: ' + row['image'])
        row['_image_sha256'] = hashlib.sha256(Path(row['image']).read_bytes()).hexdigest()
        rows.append(row)
    if not rows:
        raise ValueError('dataset is empty')
    return rows


def check_split(train, val):
    if {r['id'] for r in train} & {r['id'] for r in val}:
        raise ValueError('train/validation id overlap')
    if {r['image'] for r in train} & {r['image'] for r in val}:
        raise ValueError('train/validation image overlap')
    if {r['_image_sha256'] for r in train} & {r['_image_sha256'] for r in val}:
        raise ValueError('train/validation duplicate image content')


def answer_labels(torch, prefix, answer):
    full = torch.cat([prefix, answer], dim=1)
    if answer.shape[1] < 1:
        raise ValueError('answer has no tokens')
    # Upstream forward does not shift targets: pair x[t] with next token x[t+1].
    inputs = full[:, :-1]
    labels = full[:, 1:].clone()
    labels[:, :prefix.shape[1] - 1] = -100
    return inputs, labels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', required=True); p.add_argument('--validation', required=True)
    p.add_argument('--output', required=True, help='New directory; only connector weights are saved')
    p.add_argument('--model', default='yifanouyang/Lumen-Vision-460M')
    p.add_argument('--revision', default='d8e4540a883a5ce45cd26d36e42d4cad58ab5aa4')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--offline', action='store_true')
    p.add_argument('--steps', type=int, default=100); p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args(argv)
    if args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error('steps and learning rate must be positive and finite')
    output = Path(args.output)
    if output.exists():
        p.error('output directory exists; choose a new directory')
    train, val = read_data(args.train), read_data(args.validation)
    check_split(train, val)
    from ask_image import load_runtime
    torch, model, processor, device = load_runtime(SimpleNamespace(**vars(args)))
    from PIL import Image, ImageOps
    from safetensors.torch import save_file
    random.seed(args.seed); torch.manual_seed(args.seed)
    model.requires_grad_(False); model.MP.requires_grad_(True)
    # Fixed 512-pixel adaptation profile reduces local smoke-training memory.
    processor.inference_max_img_size = 512
    model.eval(); model.MP.train()
    optimizer = torch.optim.AdamW(model.MP.parameters(), lr=args.learning_rate, weight_decay=0.01)
    output.mkdir(parents=True)

    def loss_for(row):
        with Image.open(row['image']) as original:
            image = ImageOps.exif_transpose(original).convert('RGB')
        encoded = processor(images=image, text=row['question'], truncation=False, return_tensors='pt')
        answer = processor.tokenizer(row['answer'], add_special_tokens=False, return_tensors='pt')['input_ids']
        eos = torch.tensor([[processor.tokenizer.eos_token_id]], dtype=answer.dtype)
        input_ids, labels = answer_labels(torch, encoded['input_ids'], torch.cat([answer, eos], dim=1))
        if input_ids.shape[1] > min(model.config.lm_max_position_embeddings, 1024):
            raise ValueError('training example exceeds 1024-token local recipe; shorten answer/question')
        _, loss = model(input_ids=input_ids.to(device), images=encoded['images'].to(device), labels=labels.to(device))
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss; aborting without publishing a checkpoint')
        return loss, int((labels != -100).sum())

    indices = list(range(len(train)))
    with (output/'metrics.jsonl').open('w') as log:
        for step in range(args.steps):
            if step % len(indices) == 0:
                random.shuffle(indices)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = loss_for(train[indices[step % len(indices)]])
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.MP.parameters(), 1.0, error_if_nonfinite=True); optimizer.step()
            record = {'step': step + 1, 'train_loss': float(loss.detach())}
            log.write(json.dumps(record)+'\n'); log.flush(); print(json.dumps(record), flush=True)
        model.eval()
        with torch.no_grad():
            total = 0.0; count = 0
            for row in val:
                loss, tokens = loss_for(row); total += float(loss) * tokens; count += tokens
        metrics = {'validation_loss': total/count, 'validation_answer_tokens': count}
        log.write(json.dumps(metrics)+'\n'); print(json.dumps(metrics), flush=True)
    save_file({k:v.detach().cpu().contiguous() for k,v in model.MP.state_dict().items()}, str(output/'projector.safetensors'))
    metadata = vars(args) | {'architecture': 'VisionPsyNano', 'trainable': 'MP only',
        'image_max_size': 512, 'context_limit': 1024, 'batch_size': 1,
        'train_examples': len(train), 'validation_examples': len(val), 'validation': metrics,
        'checkpoint_type': 'projector-only; requires original base', 'weight_decay': 0.01}
    (output/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print('Saved projector-only checkpoint. Base weights are required; no accuracy claim is implied.')


if __name__ == '__main__':
    main()
