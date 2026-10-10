# Local use

Python 3.11+; around 2.03 GB for weights, plus packages and runtime memory. CPU works; CUDA is optional. The maintainer smoke test uses Python 3.13, torch 2.13.0 and Transformers 5.13.0. Other combinations are unverified.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/ask_image.py --doctor
```

Windows PowerShell: replace the activation line with `.venv\Scripts\Activate.ps1`. No API key is needed for the public checkpoint. `--doctor` checks installed packages; it does not validate GPU availability or load weights.

## One image

```bash
python scripts/ask_image.py --image photo.jpg --device cpu
```

`--device auto` chooses CUDA when available, otherwise CPU. Explicit `mps` is experimental and untested. EXIF orientation is corrected before inference. Prompts and outputs are primarily English.

## Folder → JSONL + HTML

```bash
python scripts/ask_image.py --folder ./photos --prompt "Describe this image." --output runs/results.jsonl --html runs/report.html
```

Processes JPG/PNG/WebP/BMP files in filename order, without recursion. Loads the model once. Each JSONL row records input path, prompt, model, revision and answer/error. Corrupt images are recorded as errors; any image error gives a nonzero exit code. Existing output files are overwritten. The standalone HTML embeds resized previews and escapes model text; open it in a browser. It contains your images and answers, so share it deliberately.

## Offline

On a connected machine:

```bash
hf download yifanouyang/Lumen-Vision-460M --revision d8e4540a883a5ce45cd26d36e42d4cad58ab5aa4 --local-dir checkpoints/lumen
```

Then:

```bash
python scripts/ask_image.py --model checkpoints/lumen --offline --image photo.jpg --device cpu
```

Copy the complete folder, not only the safetensors file. The CLI uses the reviewed implementation in `hf-inference/`, loads the bundled tokenizer, and does not execute Python downloaded from the Hub. The default Hub revision is pinned. If selecting another compatible Hub model, supply its own `--revision`; compatibility is unverified.

## Troubleshooting

- Dependency or ABI errors: use a fresh virtual environment and the top-level requirements.
- CUDA unavailable: use `--device cpu`; inspect your PyTorch CUDA installation separately.
- Offline missing-file error: repeat the complete download while online.
- Context/image-token error: shorten the prompt; image embeddings cannot be silently truncated.
- Low-quality answer: this is an English, compact, single-image model; results can be wrong.
