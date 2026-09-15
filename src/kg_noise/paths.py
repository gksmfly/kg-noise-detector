"""Project-wide path constants — single source of truth for every
`scripts/NN_*.py` file, instead of each one re-deriving `ROOT` from
`Path(__file__).resolve().parent.parent`.
"""
from pathlib import Path

# src/kg_noise/paths.py -> src/kg_noise -> src -> repo root
ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
ERRORS_DIR = DATA_DIR / "errors"

MODELS_DIR = ROOT / "models"
MODEL_OUT_DIR = MODELS_DIR / "kg_noise_classifier"
MODEL_BEST_DIR = MODEL_OUT_DIR / "best"

MAX_LENGTH_RECOMMENDATION_PATH = SPLITS_DIR / "max_length_recommendation.json"

# DART WikiSQL/WikiTableText domain-generalization probe (scripts 11-16) —
# relation names here are natural-language-ish table column headers
# (COLLEGE, CITY, ...) rather than WebNLG's camelCase KB identifiers, so it
# tests whether the entity/relation detection gap is a property of WebNLG's
# relation-naming style or something more general.
DART_MODEL_OUT_DIR = MODELS_DIR / "kg_noise_classifier_dart"
DART_MODEL_BEST_DIR = DART_MODEL_OUT_DIR / "best"
DART_MAX_LENGTH_RECOMMENDATION_PATH = SPLITS_DIR / "dart_max_length_recommendation.json"
