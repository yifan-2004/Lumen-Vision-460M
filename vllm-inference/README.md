# VisionPsyNano — vLLM inference

Run **VisionPsyNano** with [vLLM](https://github.com/vllm-project/vllm) through an
out-of-tree model plugin ([`visionpsy_vllm/`](visionpsy_vllm/)).

---

## Quick start

### 1. Setup

Requires Python 3.10+ and a CUDA GPU.

```bash
./scripts/setup.sh
source .venv/bin/activate
```

This installs vLLM and the plugin; the plugin is auto-registered with
`vllm serve` / `vllm.LLM` via the `vllm.general_plugins` entry point.

### 2. Run inference

```bash
bash scripts/infer.sh
```

Loads the released checkpoint (`qvac/VisionPsy-Nano-460M`) straight from the
Hugging Face Hub. Defaults: `img.jpg`, prompt *"Please describe the image."*.
Point `CKPT` at another repo id or a local folder in the same format:

```bash
CKPT=qvac/VisionPsy-Nano-460M-Flash bash scripts/infer.sh
```

Expected output: a short image description printed under `---- model output ----`.

---

## Python API

With the plugin installed, the checkpoint works with the regular vLLM Python API —
no extra wiring:

```python
from vllm import LLM

llm = LLM(model="qvac/VisionPsy-Nano-460M", dtype="float32")  # plugin auto-registers the architecture
```

Multimodal prompts must carry the image tiles and position tokens in the layout the
model was trained with; [`infer.py`](infer.py) is the complete runnable example
(image tiling → prompt construction → `llm.generate`):

```bash
python infer.py --image img.jpg --prompt "Please describe the image."
```

---

## API server

```bash
bash scripts/serve.sh
```

Starts an OpenAI-compatible server (`/v1/chat/completions`) on port 8900 with model
name `visionpsy-nano`. The server expects **pre-tiled** images — one 512×512 tile per
`image_url` item and a prompt that already carries the global/tile position tokens
(one image placeholder per tile). See [`infer.py`](infer.py) for the reference
client-side preprocessing.
