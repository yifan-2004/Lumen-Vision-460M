# Scientific model illustration

![Scientific architecture plate](../assets/model-overview.svg)

The illustration follows the implementation, rather than introducing a new architecture. A 512×512 tile becomes 32×32 visual features with hidden size 768. The actual `pixel_shuffle` reshapes each 4×4 neighborhood into 16 channel groups, producing 8×8×12288 features. A bias-free linear projection (PyTorch weight shape 960×12288) produces 64 tokens per tile. Total image-token count depends on the tiling profile.

The decoder panel expands the implemented pre-norm residual GQA and SwiGLU paths: 32 blocks, hidden 960, 15 query heads, 5 KV heads, head dimension 64. Visual features replace image-token embeddings in the causal stream; this is not a cross-attention fusion module. Frozen icons and purple gradients refer specifically to the local connector-only adaptation recipe.

The supervision panel shows the complete token stream and effective shifted labels. The last EOS is excluded from the training input, while remaining a prediction target. Prompt targets are ignored. Tensor cells and probability bars are schematic; they are not extracted attention weights or a quantitative result.

Architecture source: [Tether AI Research / VisionPsy Nano](https://github.com/tether-ai-research/qvac-visionpsy-nano). Exact tensor operations: `hf-inference/models/modality_projector.py`, `language_model.py`, and `vision_language_model.py`; supervision: `scripts/train_projector.py`. The small input image is the upstream example already present at `hf-inference/data/img.jpg`.

Regenerate the SVG with `python scripts/draw_figure.py` (requires Pillow for the embedded example thumbnail). On macOS export the PNG with `sips -s format png assets/model-overview.svg --out assets/model-overview.png`; any standards-compliant SVG renderer may be used elsewhere. SVG remains editable and scalable. `assets/model-layout.json` records verified dimensions and panel semantics; the actual layout is implemented in the renderer.
