"""Hub packaging helpers (local only — not uploaded to the model repo)."""
from __future__ import annotations

import os
import re
import shutil
from typing import Iterable, List, Tuple

AUTO_MAP = {
    "AutoConfig": "configuration_visionpsynano.VisionPsyNanoConfig",
    "AutoModel": "modeling_visionpsynano.VisionPsyNanoForConditionalGeneration",
    "AutoModelForImageTextToText": "modeling_visionpsynano.VisionPsyNanoForConditionalGeneration",
    "AutoProcessor": "processing_visionpsynano.VisionPsyNanoProcessor",
}

HUB_ROOT_FILES = [
    "configuration_visionpsynano.py",
    "modeling_visionpsynano.py",
    "processing_visionpsynano.py",
    "runtime_profile.py",
]

FLAT_VENDOR_FILES: Tuple[Tuple[str, str], ...] = (
    ("models/config.py", "vlm_config.py"),
    ("models/utils.py", "model_utils.py"),
    ("models/language_model.py", "language_model.py"),
    ("models/modality_projector.py", "modality_projector.py"),
    ("models/vision_transformer.py", "vision_transformer.py"),
    ("models/vision_language_model.py", "vision_language_model.py"),
    ("data/processors.py", "processors.py"),
    ("data/custom_transforms.py", "custom_transforms.py"),
)


def ecosystem_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _strip_vendor_bootstrap(text: str) -> str:
    text = re.sub(
        r"try:\n"
        r"    from \._vendor import ensure_vendor\n"
        r"except ImportError:\n"
        r"    from _vendor import ensure_vendor\n"
        r"\n"
        r"ensure_vendor\(__file__\)\n\n?",
        "",
        text,
    )
    text = re.sub(
        r"\n_ROOT = os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n"
        r"if _ROOT not in sys\.path:\n"
        r"    sys\.path\.insert\(0, _ROOT\)\n",
        "\n",
        text,
    )
    if "sys.path" not in text and "sys." not in text:
        text = re.sub(r"\nimport sys\n", "\n", text, count=1)
    return text


def _rewrite_for_flat_hub(text: str) -> str:
    text = _strip_vendor_bootstrap(text)

    replacements = [
        (r"from models\.config import", "from .vlm_config import"),
        (r"from models\.utils import", "from .model_utils import"),
        (r"from models\.language_model import", "from .language_model import"),
        (r"from models\.modality_projector import", "from .modality_projector import"),
        (r"from models\.vision_transformer import", "from .vision_transformer import"),
        (r"from models\.vision_language_model import", "from .vision_language_model import"),
        (r"from data\.processors import", "from .processors import"),
        (r"from data\.custom_transforms import", "from .custom_transforms import"),
        (r"from \.\.data\.processors import", "from .processors import"),
        (r"from \.config import", "from .vlm_config import"),
        (r"from \.utils import", "from .model_utils import"),
        (r"from \.custom_transforms import", "from .custom_transforms import"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    def _collapse_identical_try(match: re.Match[str]) -> str:
        body = match.group("body")
        return "".join(line[4:] if line.startswith("    ") else line for line in body.splitlines(True))

    text = re.sub(
        r"try:\n"
        r"(?P<body>(?:    from \.[^\n]+\n)+)"
        r"except ImportError:\n"
        r"(?P=body)",
        _collapse_identical_try,
        text,
    )
    return text


def copy_trust_remote_code(dst: str, *, root: str | None = None) -> List[str]:
    """Copy and flatten trust_remote_code sources into a Hub package folder."""
    root = root or ecosystem_root()
    os.makedirs(dst, exist_ok=True)
    copied: List[str] = []

    for legacy in ("models", "data", "_vendor.py"):
        legacy_path = os.path.join(dst, legacy)
        if os.path.isdir(legacy_path):
            shutil.rmtree(legacy_path)
        elif os.path.isfile(legacy_path):
            os.remove(legacy_path)

    pending: List[Tuple[str, str]] = []
    for name in HUB_ROOT_FILES:
        src = os.path.join(root, name)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing required Hub code file: {src}")
        pending.append((src, name))

    for rel_src, flat_name in FLAT_VENDOR_FILES:
        src = os.path.join(root, rel_src)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Missing required Hub code file: {src}")
        pending.append((src, flat_name))

    for src, name in pending:
        text = open(src, encoding="utf-8").read()
        text = _rewrite_for_flat_hub(text)
        out_path = os.path.join(dst, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
        copied.append(name)

    return copied


def iter_code_paths(root: str | None = None) -> Iterable[str]:
    root = root or ecosystem_root()
    for name in HUB_ROOT_FILES:
        yield os.path.join(root, name)
    for rel_src, _flat in FLAT_VENDOR_FILES:
        yield os.path.join(root, rel_src)
