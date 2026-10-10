"""Offline checks for this deployment edition; does not run the model."""
import ast
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
files = list((root / "hf-inference").rglob("*.py")) + list((root / "scripts").glob("*.py"))
for path in files:
    ast.parse(path.read_text())
manifest = json.loads((root / "docs/PROVENANCE.json").read_text())
assert manifest["training_performed"] is False
assert manifest["weights_mirror"] == "yifanouyang/Lumen-Vision-460M"
assert manifest["mirrored_large_files"]["model.safetensors"]["bytes"] == 2029498200
assert (root / "LICENSE.txt").is_file()
assert (root / "docs/UPSTREAM_README.md").is_file()
print(f"Publication checks passed; {len(files)} Python files parsed. Inference not tested.")
