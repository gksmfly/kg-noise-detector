"""kg_noise: shared library code for the kg-noise-lite pipeline.

Everything here is imported by more than one of the numbered scripts in
`scripts/` (path constants, JSONL I/O, label encoding, classification
metrics, and inference helpers for the fine-tuned classifier). Each
`scripts/NN_*.py` file stays a thin, standalone entry point that orchestrates
calls into this package rather than redefining the same helpers.
"""
