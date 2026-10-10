# Lumen Vision 460M

![Model architecture and training workflow](assets/model-overview.svg)

A compact English single-image VLM for image questions and document experiments. About **460M parameters**, with a practical local workflow: **folder → answers → JSONL + HTML report**.

[Weights](https://huggingface.co/yifanouyang/Lumen-Vision-460M) · [Local / offline setup](docs/LOCAL.md) · [Training](docs/TRAINING.md) · [Changes](docs/CHANGES.md)

## Quick start

Python 3.11+, CPU or CUDA. Run from the repository root:

```bash
pip install -r requirements.txt
python scripts/ask_image.py --doctor
python scripts/ask_image.py --image photo.jpg --device cpu
python scripts/ask_image.py --folder photos --output runs/results.jsonl --html runs/report.html
```

The CLI pins the checkpoint revision, handles image orientation, and checks image-token alignment. Cache the complete checkpoint before using `--offline`. No paid API is required.

## Train a connector

Prepare image/question/answer JSONL files with separate training and validation scenes:

```bash
python scripts/train_projector.py --train train.jsonl --validation validation.jsonl --output runs/connector --steps 100 --device cuda
python scripts/ask_image.py --image photo.jpg --projector runs/connector/projector.safetensors --device cuda
```

Both backbones stay frozen; only the visual connector is updated. See the [recipe](docs/TRAINING.md) for data format, settings and saved artifacts. CPU inference and a one-step CPU connector smoke run were checked; CUDA training is unverified. Released base weights are unchanged.

English-first, single-image input. Answers can contain hallucinations or counting errors.

## License

[Apache-2.0](LICENSE.txt) · [Source](https://github.com/tether-ai-research/qvac-visionpsy-nano) · [Model reference](https://huggingface.co/qvac/VisionPsy-Nano-460M)
