"""
Step 16 (도메인 일반화 탐침): Step 6과 동일한 TF-IDF + 코사인 유사도 베이스라인을
DART WikiSQL/WikiTableText에 다시 적합(fit)한다.

WebNLG 결과(Step 6)에서 TF-IDF는 relation corruption에 거의 무방비였다(recall
58.17%, 코사인 유사도로 relation corruption과 positive가 거의 안 갈림). DART의
relation은 표 컬럼명이라 문장에 더 잘 나타날 것으로 예상되므로, 이 베이스라인이
DART에서는 relation corruption을 더 잘 잡아내는지가 핵심 확인 포인트다.
"""
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support

from kg_noise.io_utils import load_jsonl
from kg_noise.metrics import classification_metrics
from kg_noise.paths import ERRORS_DIR, SPLITS_DIR


def load_split(name: str) -> list[dict]:
    return load_jsonl(SPLITS_DIR / f"dart_{name}.jsonl")


def similarities(vectorizer, rows: list[dict]) -> np.ndarray:
    triple_vecs = vectorizer.transform([r["triple_text"] for r in rows])
    sent_vecs = vectorizer.transform([r["sentence"] for r in rows])
    sims = np.array([
        cosine_similarity(triple_vecs[i], sent_vecs[i])[0, 0] for i in range(len(rows))
    ])
    return sims


def labels_of(rows: list[dict]) -> np.ndarray:
    return np.array([r["label"] for r in rows])


def main():
    train_rows = load_split("train")
    val_rows = load_split("val")
    test_rows = load_split("test")

    corpus = [r["triple_text"] for r in train_rows] + [r["sentence"] for r in train_rows]
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2)).fit(corpus)

    val_sims = similarities(vectorizer, val_rows)
    val_labels = labels_of(val_rows)

    best_t, best_f1 = 0.0, -1.0
    grid_log = []
    for t in np.arange(0.0, 1.01, 0.01):
        preds = (val_sims >= t).astype(int)
        f1 = precision_recall_fscore_support(
            val_labels, preds, average="macro", zero_division=0
        )[2]
        grid_log.append((float(t), float(f1)))
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"[16] validation grid search 최적 threshold={best_t:.2f} (F1_macro={best_f1:.4f})")

    test_sims = similarities(vectorizer, test_rows)
    test_labels = labels_of(test_rows)
    test_preds = (test_sims >= best_t).astype(int)
    test_metrics = classification_metrics(test_labels, test_preds)
    print(f"[16] test 성능: {test_metrics}")

    ERRORS_DIR.mkdir(parents=True, exist_ok=True)

    with (ERRORS_DIR / "dart_tfidf_baseline_predictions.jsonl").open("w", encoding="utf-8") as f:
        for r, sim, pred in zip(test_rows, test_sims, test_preds):
            f.write(json.dumps({
                "pair_id": r["pair_id"],
                "triple_text": r["triple_text"],
                "sentence": r["sentence"],
                "category": r["category"],
                "corruption_type": r["corruption_type"],
                "true_label": r["label"],
                "pred_label": int(pred),
                "cosine_similarity": float(sim),
            }, ensure_ascii=False) + "\n")

    with (ERRORS_DIR / "dart_tfidf_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "best_threshold": float(best_t),
            "val_f1_macro": float(best_f1),
            "test": test_metrics,
            "grid_log": grid_log,
        }, f, ensure_ascii=False, indent=2)

    print(f"[16] 저장 -> {ERRORS_DIR}/dart_tfidf_baseline_predictions.jsonl, dart_tfidf_metrics.json")


if __name__ == "__main__":
    main()
