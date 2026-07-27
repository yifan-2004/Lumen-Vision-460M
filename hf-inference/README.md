# VisionPsyNano — HuggingFace Transformers

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image

repo = "qvac/VisionPsy-Nano-460M"  # or qvac/VisionPsy-Nano-460M-Flash

model = AutoModelForImageTextToText.from_pretrained(
    repo, trust_remote_code=True, dtype="auto"
).cuda().eval()
model.apply_deploy_profile(model.device)

processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
inputs = processor(
    images=Image.open("data/img.jpg"),
    text="What is in this image?",
    return_tensors="pt",
)
inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items() if v is not None}
inputs.pop("pixel_values", None)

out = model.generate(**inputs, max_new_tokens=128, greedy=True)
print(processor.batch_decode(out, skip_special_tokens=True)[0])
```

```bash
python usage_inference.py --deploy --device cuda          # VisionPsyNano
python usage_inference_flash.py --deploy --device cuda    # VisionPsyNano Flash
```

Requires Python ≥ 3.10, `transformers>=4.46` (tested 5.13.1), PyTorch ≥ 2.4 + CUDA.
