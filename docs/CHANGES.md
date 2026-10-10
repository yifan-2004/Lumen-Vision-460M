# Deployment changes and validation

Original code and weights: see [provenance](PROVENANCE.json). Weights are unchanged.

- Added local, revision-pinned CLI with bundled tokenizer, explicit CPU/CUDA selection, offline mode, folder inference and self-contained HTML/JSONL exports.
- Fixed duplicate tokenizer kwargs in the original processor: explicit `padding`, `truncation` or `max_length` caused `TypeError`.
- Fixed silent visual-token mismatch: PyTorch masked scatter accepted excess visual embeddings and silently discarded them. Mismatches now produce a clear error; noncontiguous embeddings use `reshape`.
- Avoided a redundant base-tokenizer fetch during local model construction. The CLI injects the local bundled tokenizer path before creating the model.

Run `python -m unittest discover -s tests -v` and `python scripts/verify_publication.py`. Numerical regressions use torch and skip if it is unavailable. Tests cover actual source methods, image discovery, CLI dependency isolation and HTML escaping. They do not establish model accuracy, CUDA/MPS performance or training compatibility.

## Actual CPU smoke run

On 2026-10-09 (America/Los_Angeles), the complete 2.03 GB checkpoint was downloaded and loaded locally. Online first load and offline cached load both completed on CPU with Python 3.13 / torch 2.13.0 / Transformers 5.13.0. Offline 24-token greedy generation on the upstream polar-bear sample produced:

> The image depicts a polar bear resting on a large, flat rock. The bear is lying down with its body stretched out

This is a deliberately short smoke output, not an accuracy benchmark. [Actual result JSON](examples/polar-bear-result.json) · [Self-contained HTML report](examples/polar-bear-report.html) (download and open locally).

CUDA/MPS, the full paper benchmarks, server backends and training were not executed. Ten regression tests passed locally, including numerical embedding-count checks.
