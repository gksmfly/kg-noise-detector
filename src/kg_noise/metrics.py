"""Classification metrics shared by the BERT trainer (script 05) and the
TF-IDF baseline (script 06) — both computed the exact same macro + noisy-
class breakdown independently, which is also why their `*_metrics.json`
outputs are directly comparable in `scripts/07_evaluate_compare.py`.
"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from kg_noise.constants import LABEL_NOISY


def classification_metrics(y_true, y_pred) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_n, r_n, f1_n, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[LABEL_NOISY], average=None, zero_division=0
    )
    return {
        "accuracy": acc,
        "precision_macro": p_macro,
        "recall_macro": r_macro,
        "f1": f1_macro,
        "recall_noisy": r_n[0],
        "precision_noisy": p_n[0],
        "f1_noisy": f1_n[0],
    }
