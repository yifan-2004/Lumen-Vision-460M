# Training Lumen Vision 460M

![Architecture and local adaptation](../assets/model-overview.svg)

## Original model training (reported by the original authors)

The released base starts from nanoVLM-460M-8k. Its published post-training sequence is broad SFT, capability-focused SFT, weakness-targeted specialists, TIES weight merging, then preference optimization with replay. Early stages develop coverage; later stages focus on errors and response reliability. This repository does not reproduce those stages or infer unpublished hyperparameters. See [Tether AI Research’s training description](https://huggingface.co/blog/qvac/visionpsy#how-we-trained-it) and [source model card](https://huggingface.co/qvac/VisionPsy-Nano-460M).

## Runnable local recipe: connector-only adaptation

1. Install [local requirements](LOCAL.md). Use images and answers whose training use you have permission for. Split by document/scene before formatting: this script rejects repeated IDs, image paths and byte-identical images across splits; it cannot detect all near duplicates.
2. Make `train.jsonl` and `validation.jsonl`. Images are relative to each JSONL file:

```json
{"id":"scene-001","image":"images/scene-001.jpg","question":"What is visible?","answer":"A red cup on a table."}
```

3. Run a one-step hardware check, then increase steps only after reviewing losses:

```bash
python scripts/train_projector.py --train train.jsonl --validation validation.jsonl --output runs/connector-check --steps 1 --device cpu
python scripts/train_projector.py --train train.jsonl --validation validation.jsonl --output runs/connector-100 --steps 100 --device cuda --learning-rate 1e-5
```

Add `--offline` after caching the checkpoint. Each output directory must be new. CUDA is not yet tested; CPU was tested with the real base checkpoint.

4. Inspect `metrics.jsonl`: per-step training loss and held-out loss weighted by answer tokens. Save artifacts are `projector.safetensors` and `training.json`. Evaluate generations on a held-out task against the unchanged base; loss alone does not establish improved VQA accuracy.
5. Load the connector for inference:

```bash
python scripts/ask_image.py --image photo.jpg --projector runs/connector-100/projector.safetensors --device cpu
```

Use the same base ID/revision recorded in `training.json`. A connector file alone is not a complete model. Inference tiling defaults differ from the 512-pixel adaptation preset; measure this distribution change before claiming improvements.

| Setting | This local recipe |
| --- | --- |
| Trainable | Pixel-shuffle linear connector `MP`; both backbones frozen |
| Objective | Causal next-token cross-entropy on answer + EOS only; prompt labels masked |
| Optimizer | AdamW, learning rate 1e-5, weight decay 0.01, gradient norm 1.0 |
| Precision / batch | FP32 / one image-question-answer per step |
| Image / context | Maximum 512 pixels before tiling / 1,024 total input tokens |
| Seed / length | 42 / explicitly supplied step count; no scheduler or resume |

The original forward expects already shifted labels; the trainer constructs those explicitly. Frozen decoder parameters still pass gradients to the connector. Memory depends on image tiles and sequence length; a full-backbone training configuration is not provided.

## Validation scope

A real 460M CPU run completed one connector update and held-out validation on two synthetic solid-color images, then saved and reloaded the connector for inference. This validates the execution path, not useful task learning or original training reproduction. No adapted weights from that toy run are published as a new model. Use the source dataset licenses when reproducing original training; they are separate from the model’s Apache-2.0 license.
