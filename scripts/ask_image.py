#!/usr/bin/env python3
"""Lumen Vision: use the published checkpoint with the reference runtime."""
import os
import runpy
from pathlib import Path
import sys
runtime = Path(__file__).resolve().parents[1] / "hf-inference"
os.environ.setdefault("VISIONPSY_NANO", "yifanouyang/Lumen-Vision-460M")
sys.path.insert(0, str(runtime))
runpy.run_path(str(runtime / "usage_inference.py"), run_name="__main__")
