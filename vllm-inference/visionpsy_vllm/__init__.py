"""Out-of-tree vLLM plugin for VisionPsyNano.

When installed with pip, vLLM auto-loads `register()` at startup through the
`vllm.general_plugins` entry point, so `vllm serve` and `vllm.LLM(...)` accept
VisionPsyNano checkpoints directly. For in-process use without installation,
import this package and call `register()` before constructing the engine.
"""


def register() -> None:
    """Register the VisionPsyNano config + model class with transformers and vLLM.

    Re-entrant: safe to call multiple times / once per process.
    """
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .configuration_visionpsynano import VisionPsyNanoConfig

    try:
        AutoConfig.register("visionpsynano", VisionPsyNanoConfig)
    except ValueError:
        # already registered in this process
        pass

    if "VisionPsyNanoForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "VisionPsyNanoForConditionalGeneration",
            "visionpsy_vllm.modeling_visionpsynano:VisionPsyNanoForConditionalGeneration",
        )
