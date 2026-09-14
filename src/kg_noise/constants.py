"""Label encoding and base model name shared across the pipeline.

label: positive (corruption_type=none) -> LABEL_GROUNDED (1),
negative (entity/relation corruption) -> LABEL_NOISY (0).
"""

MODEL_NAME = "bert-base-cased"
LABEL_NOISY, LABEL_GROUNDED = 0, 1
