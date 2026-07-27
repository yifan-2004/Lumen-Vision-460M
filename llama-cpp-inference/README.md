# VisionPsyNano — llama.cpp inference

Run **VisionPsyNano** (GGUF) with a patched [llama.cpp](llama.cpp-custom/) fork.

## Quick start

### 1. Download weights

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

```bash
GG="${GG:-$HOME/checkpoints/visionpsy-quantized-models}"
NANO="$GG/VisionPsy-Nano-460M"
FLASH_D="$GG/VisionPsy-Nano-460M-Flash"
```

Prefix any command with `BACKEND=cpu` to run on CPU.

### VisionPsyNano

```bash
LM="$NANO/VisionPsy-Nano-460M.gguf" MMPROJ="$NANO/VisionPsy-Nano-460M-mmproj.q8_0.gguf" bash scripts/infer.sh
LM="$NANO/VisionPsy-Nano-460M.bf16.gguf" MMPROJ="$NANO/VisionPsy-Nano-460M-mmproj.q8_0.gguf" bash scripts/infer.sh
LM="$NANO/VisionPsy-Nano-460M.f16.gguf" MMPROJ="$NANO/VisionPsy-Nano-460M-mmproj.q8_0.gguf" bash scripts/infer.sh
LM="$NANO/VisionPsy-Nano-460M.q4_0.gguf" MMPROJ="$NANO/VisionPsy-Nano-460M-mmproj.q8_0.gguf" bash scripts/infer.sh
LM="$NANO/VisionPsy-Nano-460M.q5_k_m.gguf" MMPROJ="$NANO/VisionPsy-Nano-460M-mmproj.q8_0.gguf" bash scripts/infer.sh
LM="$NANO/VisionPsy-Nano-460M.q8_0.gguf" MMPROJ="$NANO/VisionPsy-Nano-460M-mmproj.q8_0.gguf" bash scripts/infer.sh
```

### VisionPsyNano Flash

```bash
LM="$FLASH_D/VisionPsy-Nano-460M-Flash.gguf" MMPROJ="$FLASH_D/VisionPsy-Nano-460M-Flash-mmproj.q8_0.gguf" FLASH=1 bash scripts/infer.sh
LM="$FLASH_D/VisionPsy-Nano-460M-Flash.bf16.gguf" MMPROJ="$FLASH_D/VisionPsy-Nano-460M-Flash-mmproj.q8_0.gguf" FLASH=1 bash scripts/infer.sh
LM="$FLASH_D/VisionPsy-Nano-460M-Flash.f16.gguf" MMPROJ="$FLASH_D/VisionPsy-Nano-460M-Flash-mmproj.q8_0.gguf" FLASH=1 bash scripts/infer.sh
LM="$FLASH_D/VisionPsy-Nano-460M-Flash.q3_k_m.gguf" MMPROJ="$FLASH_D/VisionPsy-Nano-460M-Flash-mmproj.q8_0.gguf" FLASH=1 bash scripts/infer.sh
LM="$FLASH_D/VisionPsy-Nano-460M-Flash.q4_0.gguf" MMPROJ="$FLASH_D/VisionPsy-Nano-460M-Flash-mmproj.q8_0.gguf" FLASH=1 bash scripts/infer.sh
LM="$FLASH_D/VisionPsy-Nano-460M-Flash.q5_k_m.gguf" MMPROJ="$FLASH_D/VisionPsy-Nano-460M-Flash-mmproj.q8_0.gguf" FLASH=1 bash scripts/infer.sh
LM="$FLASH_D/VisionPsy-Nano-460M-Flash.q8_0.gguf" MMPROJ="$FLASH_D/VisionPsy-Nano-460M-Flash-mmproj.q8_0.gguf" FLASH=1 bash scripts/infer.sh
```
