# Training scope

This release provides inference tools, not a validated training pipeline. No new weights were trained.

For domain adaptation, use the [original nanoVLM training project](https://github.com/huggingface/nanoVLM), review its checkpoint-conversion path and dataset licenses, and reserve a held-out evaluation split before training. The original model card records architecture, data and evaluation details.

Export a separate checkpoint and document data, hardware, hyperparameters and before/after metrics. Do not overwrite the mirrored release or assume its tokenizer/checkpoint format is directly accepted by another trainer. No finetuning command is advertised here without a verified end-to-end run.
