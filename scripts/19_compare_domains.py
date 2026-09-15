"""
Step 19 (도메인 일반화 탐침, 종합): WebNLG(본실험)와 DART WikiSQL/WikiTableText
(도메인 일반화 탐침) 결과를 한 리포트로 묶는다.

원래 가설: DART는 relation 이름이 표 컬럼명이라 WebNLG의 카멜케이스 KB 식별자보다
자연어에 훨씬 가깝다 — 그러니 entity/relation 탐지 격차가 DART에서는 좁혀질 것으로
예상했다.

실제 결과는 정반대다: BERT 기준 격차가 WebNLG 1.57%p -> DART 28.73%p로 훨씬
커졌다. 이 스크립트는 그 반전을 수치로 정리하고, "relation 이름의 자연어성"과
"데이터 규모/relation당 예시 수"라는 두 요인이 뒤섞여 있다는 교란 요인을 명시적으로
드러낸다(하나만 바꾼 통제 실험이 아니라 도메인을 통째로 바꿨기 때문).
"""
import json

from kg_noise.io_utils import load_jsonl
from kg_noise.paths import ERRORS_DIR, PROCESSED_DIR, RAW_DIR, SPLITS_DIR

WEBNLG_BERT = {"entity": 0.9965, "relation": 0.9808}
WEBNLG_TFIDF = {"entity": 0.7694, "relation": 0.5817}


def corpus_stats(raw_path, train_path):
    raw_rows = load_jsonl(raw_path)
    train_rows = load_jsonl(train_path)
    n_relations = len(set(r["relation"] for r in raw_rows))
    return {
        "n_positive": len(raw_rows),
        "n_train_rows": len(train_rows),
        "n_unique_relations": n_relations,
        "examples_per_relation": len(raw_rows) / n_relations,
    }


def main():
    webnlg_stats = corpus_stats(RAW_DIR / "webnlg_pairs.jsonl", SPLITS_DIR / "train.jsonl")
    dart_stats = corpus_stats(RAW_DIR / "dart_wikitable_pairs.jsonl", SPLITS_DIR / "dart_train.jsonl")

    dart_bert_test = json.loads((ERRORS_DIR / "dart_bert_metrics.json").read_text())["test"]
    dart_tfidf_test = json.loads((ERRORS_DIR / "dart_tfidf_metrics.json").read_text())["test"]

    dart_bert_rows = load_jsonl(ERRORS_DIR / "dart_bert_test_predictions.jsonl")
    dart_tfidf_rows = load_jsonl(ERRORS_DIR / "dart_tfidf_baseline_predictions.jsonl")

    def recall_by_type(rows):
        from collections import defaultdict
        by_type = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["corruption_type"] == "none":
                continue
            by_type[r["corruption_type"]][1] += 1
            if r["pred_label"] == 0:
                by_type[r["corruption_type"]][0] += 1
        return {t: c / n for t, (c, n) in by_type.items()}

    dart_bert_recall = recall_by_type(dart_bert_rows)
    dart_tfidf_recall = recall_by_type(dart_tfidf_rows)

    lines = [
        "# Step 19: WebNLG vs DART(WikiSQL/WikiTableText) 도메인 일반화 비교\n",
        "## 가설과 반대로 나온 핵심 결과\n",
        "relation 이름이 WebNLG의 카멜케이스 KB 식별자(`cityServed`)보다 자연어에 훨씬 "
        "가까운 DART 표 컬럼명(`COLLEGE`, `CITY`)을 썼을 때, entity/relation 탐지 "
        "격차가 좁혀질 것으로 예상했다. **실제로는 정반대로 훨씬 크게 벌어졌다.**\n",
        "| 도메인 | 모델 | Entity corruption recall | Relation corruption recall | Gap |",
        "|---|---|---|---|---|",
        f"| WebNLG | BERT (fine-tuned) | {WEBNLG_BERT['entity']:.4f} | {WEBNLG_BERT['relation']:.4f} | {WEBNLG_BERT['entity']-WEBNLG_BERT['relation']:+.4f} |",
        f"| WebNLG | TF-IDF baseline | {WEBNLG_TFIDF['entity']:.4f} | {WEBNLG_TFIDF['relation']:.4f} | {WEBNLG_TFIDF['entity']-WEBNLG_TFIDF['relation']:+.4f} |",
        f"| DART | BERT (fine-tuned) | {dart_bert_recall['entity']:.4f} | {dart_bert_recall['relation']:.4f} | {dart_bert_recall['entity']-dart_bert_recall['relation']:+.4f} |",
        f"| DART | TF-IDF baseline | {dart_tfidf_recall['entity']:.4f} | {dart_tfidf_recall['relation']:.4f} | {dart_tfidf_recall['entity']-dart_tfidf_recall['relation']:+.4f} |",
        "",
        "## 데이터 규모 비교 — 교란 요인 후보\n",
        "DART로 도메인만 바꾼 게 아니라, 표본 크기와 relation당 예시 수도 함께 크게 "
        "줄었다. 이 두 요인이 뒤섞여 있어 '자연어성' 하나만의 효과라고 단정할 수 없다.\n",
        "| | WebNLG | DART (WikiSQL/WikiTableText) | 비율 |",
        "|---|---|---|---|",
        f"| positive 샘플 수 | {webnlg_stats['n_positive']:,} | {dart_stats['n_positive']:,} | {dart_stats['n_positive']/webnlg_stats['n_positive']*100:.0f}% |",
        f"| train 행 수 | {webnlg_stats['n_train_rows']:,} | {dart_stats['n_train_rows']:,} | {dart_stats['n_train_rows']/webnlg_stats['n_train_rows']*100:.0f}% |",
        f"| 고유 relation 수 | {webnlg_stats['n_unique_relations']:,} | {dart_stats['n_unique_relations']:,} | {dart_stats['n_unique_relations']/webnlg_stats['n_unique_relations']*100:.0f}% |",
        f"| relation당 평균 예시 수 | {webnlg_stats['examples_per_relation']:.1f} | {dart_stats['examples_per_relation']:.1f} | {dart_stats['examples_per_relation']/webnlg_stats['examples_per_relation']*100:.0f}% |",
        "",
        "## 해석\n",
        "가능한 설명 두 가지가 뒤섞여 있다:\n",
        "1. **자연어성 가설이 틀렸다**: relation 이름이 자연어에 가까워도 recall 격차를 "
        "좁히지 못한다 — 표면 어휘 중첩과 무관하게 relation 검증 자체가 entity 검증보다 "
        "근본적으로 더 어려운 과제일 수 있다.",
        "2. **데이터 희소성이 주된 원인**: DART는 relation당 평균 예시 수가 WebNLG의 "
        "약 10분의 1이다. Entity 판정(고유 문자열이 다른지 보는 국소적 판단)은 적은 "
        "데이터로도 배우기 쉽지만, relation 판정(트리플과 문장의 의미적 정합성 판단)은 "
        "더 많은 반복 노출이 필요한 과제라서, 데이터가 줄면 relation recall이 "
        "불균형하게 더 크게 떨어질 수 있다.",
        "",
        "이 둘을 가르는 ablation: WebNLG train을 DART와 같은 표본 크기(4,968행)로 "
        "무작위 서브샘플링해서 Step 5와 동일하게 재학습하면 된다. relation 어휘의 "
        "자연어성은 WebNLG 그대로 유지한 채 표본 크기만 맞추는 대조군이므로, 거기서도 "
        "relation recall이 크게 떨어지면 (2)가 주된 원인이고, 여전히 격차가 작게 "
        "유지되면 (1)에 더 무게가 실린다.",
    ]

    out_path = ERRORS_DIR / "domain_comparison_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[19] 저장 -> {out_path}")


if __name__ == "__main__":
    main()
