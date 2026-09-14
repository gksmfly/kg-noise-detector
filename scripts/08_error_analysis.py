"""
Step 8: 핵심 연구 질문 검증 — entity_corruption과 relation_corruption의 탐지 난이도가
다른가?

가설: entity corruption은 표면적 어휘 불일치로 나타나 탐지가 쉽고(예: Seoul->Busan),
relation corruption은 어휘 중첩이 높아 의미적으로만 다르므로 탐지가 구조적으로 더
어렵다(예: capital->largest_city). 트리플 직렬화("subject | relation | object")에서
relation 슬롯의 카멜케이스 식별자(예: cityServed)는 애초에 문장 표면에 거의 등장하지
않으므로, TF-IDF처럼 표면 어휘 중첩에만 의존하는 baseline은 relation corruption 여부와
무관하게 코사인 유사도가 비슷하게 나와 사실상 탐지를 못할 가능성이 있다. 이걸 BERT와
TF-IDF 두 모델 모두에서 entity vs relation corruption별 recall을 분리 계산해 검증한다.
"""
from collections import Counter, defaultdict

from kg_noise.io_utils import load_jsonl
from kg_noise.paths import ERRORS_DIR


def recall_by_corruption(rows: list[dict]) -> dict:
    """corruption_type(entity/relation)별로, 그 negative들 중 몇 %가 실제로
    noisy(0)로 올바르게 예측됐는지(recall) 계산."""
    by_type = defaultdict(lambda: [0, 0])  # type -> [correct, total]
    for r in rows:
        if r["corruption_type"] == "none":
            continue
        by_type[r["corruption_type"]][1] += 1
        if r["pred_label"] == 0:  # noisy로 올바르게 예측
            by_type[r["corruption_type"]][0] += 1
    return {t: (c / n if n else 0.0, c, n) for t, (c, n) in by_type.items()}


def main():
    bert_rows = load_jsonl(ERRORS_DIR / "bert_test_predictions.jsonl")
    tfidf_rows = load_jsonl(ERRORS_DIR / "tfidf_baseline_predictions.jsonl")

    bert_recall = recall_by_corruption(bert_rows)
    tfidf_recall = recall_by_corruption(tfidf_rows)

    print("[8] === entity vs relation corruption 탐지 recall (핵심 결과) ===")
    for model_name, recall in [("BERT", bert_recall), ("TF-IDF", tfidf_recall)]:
        for t in ["entity", "relation"]:
            rate, c, n = recall.get(t, (0.0, 0, 0))
            print(f"[8] {model_name:7s} {t:9s} recall = {rate:.4f} ({c}/{n})")

    # TF-IDF cosine similarity로 가설의 메커니즘(표면 어휘 중첩)을 직접 들여다본다.
    sim_by_type = defaultdict(list)
    for r in tfidf_rows:
        sim_by_type[r["corruption_type"]].append(r["cosine_similarity"])
    sim_stats = {}
    for t, sims in sim_by_type.items():
        sim_stats[t] = {"mean": sum(sims) / len(sims), "n": len(sims)}
    print(f"[8] TF-IDF cosine similarity 평균 (corruption_type별): "
          f"{ {t: round(v['mean'], 4) for t, v in sim_stats.items()} }")

    # BERT 오분류 상세 (noisy인데 grounded로 오판, 확신도 높은 순, corruption_type별로)
    bert_by_id = {r["pair_id"] + ":" + r["corruption_type"]: r for r in bert_rows}
    fn_by_type = defaultdict(list)
    for r in bert_rows:
        if r["true_label"] == 0 and r["pred_label"] == 1:
            fn_by_type[r["corruption_type"]].append(r)
    for t in fn_by_type:
        fn_by_type[t].sort(key=lambda r: -r["prob_grounded"])

    cat_counter_all = Counter(r["category"] for r in bert_rows)
    cat_counter_fn = Counter(r["category"] for t in fn_by_type for r in fn_by_type[t])

    report = [
        "# Step 8: KG-to-Text 라벨 노이즈 탐지 - entity vs relation corruption 난이도 비교\n",
        "## 핵심 결과: corruption_type별 recall\n",
        "| Model | Entity corruption recall | Relation corruption recall | Gap |",
        "|---|---|---|---|",
    ]
    for model_name, recall in [("BERT (fine-tuned)", bert_recall), ("TF-IDF baseline", tfidf_recall)]:
        e_rate, e_c, e_n = recall.get("entity", (0.0, 0, 0))
        r_rate, r_c, r_n = recall.get("relation", (0.0, 0, 0))
        gap = e_rate - r_rate
        report.append(
            f"| {model_name} | {e_rate:.4f} ({e_c}/{e_n}) | {r_rate:.4f} ({r_c}/{r_n}) | "
            f"{gap:+.4f} |"
        )

    report.append("\n## TF-IDF cosine similarity 평균 (corruption_type별)\n")
    report.append("이 값이 낮을수록 negative를 쉽게 걸러낼 수 있다는 뜻이다 (threshold보다 "
                   "낮게 떨어지므로). positive와 비슷하게 높다면 표면 어휘로는 구분이 "
                   "안 된다는 뜻이다.\n")
    report.append("| corruption_type | mean cosine similarity | n |")
    report.append("|---|---|---|")
    for t in ["none", "entity", "relation"]:
        if t in sim_stats:
            report.append(f"| {t} | {sim_stats[t]['mean']:.4f} | {sim_stats[t]['n']} |")

    hyp_supported = (
        bert_recall.get("entity", (0,))[0] > bert_recall.get("relation", (0,))[0]
        and tfidf_recall.get("entity", (0,))[0] > tfidf_recall.get("relation", (0,))[0]
    )
    report.append(f"\n## 가설 검증 결과\n")
    report.append(
        f"가설(\"entity corruption이 relation corruption보다 탐지하기 쉽다\")은 "
        f"BERT와 TF-IDF 모두에서 {'지지됨' if hyp_supported else '지지되지 않음 (재확인 필요)'}."
    )

    report.append("\n## BERT: relation_corruption을 grounded로 오판한 사례 (확신도 높은 순, 최대 15건)\n")
    for i, r in enumerate(fn_by_type.get("relation", [])[:15]):
        report.append(f"### 사례 {i + 1} (pair_id={r['pair_id']}, P(grounded)={r['prob_grounded']:.3f})")
        report.append(f"- triple_text: {r['triple_text']}")
        report.append(f"- sentence: {r['sentence']}")
        report.append("")

    report.append("\n## BERT: entity_corruption을 grounded로 오판한 사례 (확신도 높은 순, 최대 15건)\n")
    for i, r in enumerate(fn_by_type.get("entity", [])[:15]):
        report.append(f"### 사례 {i + 1} (pair_id={r['pair_id']}, P(grounded)={r['prob_grounded']:.3f})")
        report.append(f"- triple_text: {r['triple_text']}")
        report.append(f"- sentence: {r['sentence']}")
        report.append("")

    n_fn_entity = len(fn_by_type.get("entity", []))
    n_fn_relation = len(fn_by_type.get("relation", []))
    report.append("\n## category별 오분류 분포 (참고)\n")
    report.append(f"- 전체 category 분포: {dict(cat_counter_all.most_common(10))}")
    report.append(f"- 오분류(noisy->grounded) category 분포: {dict(cat_counter_fn.most_common(10))}")
    report.append(f"- BERT entity corruption 오분류 {n_fn_entity}건 / relation corruption 오분류 {n_fn_relation}건")

    out_path = ERRORS_DIR / "error_analysis_report.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[8] 상세 리포트 저장 -> {out_path}")


if __name__ == "__main__":
    main()
