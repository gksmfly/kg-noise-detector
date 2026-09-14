"""
Step 7: Test set에서 BERT fine-tuned 모델과 TF-IDF baseline 성능을 표 하나로 비교.
"""
import json

from kg_noise.paths import ERRORS_DIR


def main():
    bert = json.loads((ERRORS_DIR / "bert_metrics.json").read_text())["test"]
    tfidf = json.loads((ERRORS_DIR / "tfidf_metrics.json").read_text())["test"]

    rows = [
        ("Accuracy", bert.get("test_accuracy", bert.get("accuracy")), tfidf["accuracy"]),
        ("Precision (macro)", bert.get("test_precision_macro", bert.get("precision_macro")), tfidf["precision_macro"]),
        ("Recall (macro)", bert.get("test_recall_macro", bert.get("recall_macro")), tfidf["recall_macro"]),
        ("F1 (macro)", bert.get("test_f1", bert.get("f1")), tfidf["f1"]),
        ("Recall (noisy)", bert.get("test_recall_noisy", bert.get("recall_noisy")), tfidf["recall_noisy"]),
        ("Precision (noisy)", bert.get("test_precision_noisy", bert.get("precision_noisy")), tfidf["precision_noisy"]),
        ("F1 (noisy)", bert.get("test_f1_noisy", bert.get("f1_noisy")), tfidf["f1_noisy"]),
    ]

    lines = ["| Metric | BERT (fine-tuned) | TF-IDF baseline |", "|---|---|---|"]
    for name, b, t in rows:
        lines.append(f"| {name} | {b:.4f} | {t:.4f} |")
    table = "\n".join(lines)
    print(table)

    out_path = ERRORS_DIR / "comparison_table.md"
    out_path.write_text(table + "\n", encoding="utf-8")
    print(f"\n[7] 저장 -> {out_path}")


if __name__ == "__main__":
    main()
