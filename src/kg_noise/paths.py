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
