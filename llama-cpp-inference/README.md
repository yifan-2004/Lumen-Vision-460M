# VisionPsyNano — llama.cpp inference

Run **VisionPsyNano** (GGUF) with a patched [llama.cpp](llama.cpp-custom/) fork.

## Quick start

### 1. Download weights

GGUF builds (fp32 down to 3-bit) are published on the Hugging Face Hub:

- [qvac/VisionPsy-Nano-460M-GGUFs](https://huggingface.co/qvac/VisionPsy-Nano-460M-GGUFs)
- [qvac/VisionPsy-Nano-460M-Flash-GGUFs](https://huggingface.co/qvac/VisionPsy-Nano-460M-Flash-GGUFs)

Download one LM build plus the vision projector (`mmproj`), and place them as
`lm.gguf` / `mmproj.gguf`. For example, the 4-bit build used for our on-device
benchmarks:

```bash
# with the Hugging Face CLI (pip install -U huggingface_hub);
# or download the same files from the repo page in a browser
huggingface-cli download qvac/VisionPsy-Nano-460M-GGUFs \
  visionpsy-nano-460m-q4_0.gguf mmproj-visionpsy-nano-460m-q8.gguf \
  --local-dir models/nano
mv models/nano/visionpsy-nano-460m-q4_0.gguf models/nano/lm.gguf
mv models/nano/mmproj-visionpsy-nano-460m-q8.gguf models/nano/mmproj.gguf
```

Flash: same commands with the `-Flash-GGUFs` repo and `models/flash/`
(file names carry a `-flash` infix, e.g. `visionpsy-nano-460m-flash-q4_0.gguf`).

Expected layout:

```
models/
├── nano/
│   ├── lm.gguf
│   └── mmproj.gguf
└── flash/
    ├── lm.gguf
    └── mmproj.gguf
```

### 2. Build

**GPU (CUDA)** — needs a NVIDIA GPU and `nvcc` (default `CUDA_ARCH=90` for H100; use `80` for A100):

```bash
./scripts/build.sh
# CUDA_ARCH=80 ./scripts/build.sh
```

Binary: `llama.cpp-custom/build_cuda/bin/llama-mtmd-cli`.

**CPU** — no CUDA toolkit required:

```bash
BACKEND=cpu ./scripts/build.sh
```

Binary: `llama.cpp-custom/build_cpu/bin/llama-mtmd-cli`.

### 3. Inference

**GPU:**

```bash
bash scripts/infer.sh                 # VisionPsyNano
FLASH=1 bash scripts/infer.sh         # VisionPsyNano Flash
```

**CPU:**

```bash
BACKEND=cpu bash scripts/infer.sh
BACKEND=cpu FLASH=1 bash scripts/infer.sh
```

Defaults: `img.jpg`, prompt *"Please describe the image."*.  
Overrides: `IMAGE=...` `PROMPT='...'` `MAX_TOKENS=256` `LM=...` `MMPROJ=...`.

---

## Other precisions

Each repo ships a full quantization ladder — `fp32`, `bf16`, `q8_0`, `q5_k_m`,
`q4_k_m-imat`, `q4_0`, and `iq4`/`iq3` imatrix variants (~1.6 GB down to ~240 MB).
See the repo file listings for exact names. To run any of them, point `LM` at the
downloaded file:

```bash
LM=models/nano/visionpsy-nano-460m-q8_0.gguf MMPROJ=models/nano/mmproj.gguf bash scripts/infer.sh
```

Prefix any command with `BACKEND=cpu` to run on CPU, and add `FLASH=1` for the
Flash model.
