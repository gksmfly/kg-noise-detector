"""
Step 18 (도메인 일반화 탐침): Step 8과 동일하게, DART 도메인에서 entity_corruption과
relation_corruption의 탐지 난이도를 분리해서 본다.

WebNLG(Step 8)에서는 BERT의 entity/relation recall 격차가 1.6%p로 작았다. DART에서
같은 절차로 다시 재는 이유는, relation 이름이 더 자연어에 가까운 이 도메인에서
격차가 좁혀지는지(가설의 메커니즘이 맞다는 증거) 확인하기 위해서였다.
"""
from collections import Counter, defaultdict

from kg_noise.io_utils import load_jsonl
from kg_noise.paths import ERRORS_DIR


def recall_by_corruption(rows: list[dict]) -> dict:
    by_type = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["corruption_type"] == "none":
            continue
        by_type[r["corruption_type"]][1] += 1
        if r["pred_label"] == 0:
            by_type[r["corruption_type"]][0] += 1
    return {t: (c / n if n else 0.0, c, n) for t, (c, n) in by_type.items()}


def main():
    bert_rows = load_jsonl(ERRORS_DIR / "dart_bert_test_predictions.jsonl")
    tfidf_rows = load_jsonl(ERRORS_DIR / "dart_tfidf_baseline_predictions.jsonl")

    bert_recall = recall_by_corruption(bert_rows)
    tfidf_recall = recall_by_corruption(tfidf_rows)

    print("[18] === DART: entity vs relation corruption 탐지 recall ===")
    for model_name, recall in [("BERT", bert_recall), ("TF-IDF", tfidf_recall)]:
        for t in ["entity", "relation"]:
            rate, c, n = recall.get(t, (0.0, 0, 0))
            print(f"[18] {model_name:7s} {t:9s} recall = {rate:.4f} ({c}/{n})")

    sim_by_type = defaultdict(list)
    for r in tfidf_rows:
        sim_by_type[r["corruption_type"]].append(r["cosine_similarity"])
    sim_stats = {}
    for t, sims in sim_by_type.items():
        sim_stats[t] = {"mean": sum(sims) / len(sims), "n": len(sims)}
    print(f"[18] TF-IDF cosine similarity 평균 (corruption_type별): "
          f"{ {t: round(v['mean'], 4) for t, v in sim_stats.items()} }")

    fn_by_type = defaultdict(list)
    for r in bert_rows:
        if r["true_label"] == 0 and r["pred_label"] == 1:
            fn_by_type[r["corruption_type"]].append(r)
    for t in fn_by_type:
        fn_by_type[t].sort(key=lambda r: -r["prob_grounded"])

    cat_counter_all = Counter(r["category"] for r in bert_rows)
    cat_counter_fn = Counter(r["category"] for t in fn_by_type for r in fn_by_type[t])

    report = [
        "# Step 18: DART(WikiSQL/WikiTableText) 도메인 일반화 탐침 - entity vs relation corruption 난이도\n",
        "**배경**: WebNLG의 relation은 카멜케이스 KB 식별자(`cityServed`)라 문장 표면에 "
        "거의 등장하지 않는다. DART의 WikiSQL/WikiTableText 서브셋은 relation이 표 "
        "컬럼명(`COLLEGE`, `CITY`, `FEET`)이라 훨씬 자연어에 가깝다. 같은 corruption "
        "방법론을 그대로 적용해, relation 이름의 자연어성이 entity/relation 탐지 격차를 "
        "좁히는지 검증한다.\n",
        "## 핵심 결과: corruption_type별 recall\n",
        "| Model | Entity corruption recall | Relation corruption recall | Gap |",
        "|---|---|---|---|",
    ]
    for model_name, recall in [("BERT (fine-tuned, DART)", bert_recall), ("TF-IDF baseline (DART)", tfidf_recall)]:
        e_rate, e_c, e_n = recall.get("entity", (0.0, 0, 0))
        r_rate, r_c, r_n = recall.get("relation", (0.0, 0, 0))
        gap = e_rate - r_rate
        report.append(
            f"| {model_name} | {e_rate:.4f} ({e_c}/{e_n}) | {r_rate:.4f} ({r_c}/{r_n}) | "
            f"{gap:+.4f} |"
        )

    report.append("\n## TF-IDF cosine similarity 평균 (corruption_type별)\n")
    report.append("| corruption_type | mean cosine similarity | n |")
    report.append("|---|---|---|")
    for t in ["none", "entity", "relation"]:
        if t in sim_stats:
            report.append(f"| {t} | {sim_stats[t]['mean']:.4f} | {sim_stats[t]['n']} |")

    report.append("\n## WebNLG(Step 8) 대비 비교\n")
    report.append("| 도메인 | 모델 | Entity recall | Relation recall | Gap |")
    report.append("|---|---|---|---|---|")
    report.append("| WebNLG | BERT | 0.9965 | 0.9808 | +0.0157 |")
    report.append("| WebNLG | TF-IDF | 0.7694 | 0.5817 | +0.1878 |")
    e_rate, e_c, e_n = bert_recall.get("entity", (0.0, 0, 0))
    r_rate, r_c, r_n = bert_recall.get("relation", (0.0, 0, 0))
    report.append(f"| DART | BERT | {e_rate:.4f} | {r_rate:.4f} | {e_rate-r_rate:+.4f} |")
    e_rate, e_c, e_n = tfidf_recall.get("entity", (0.0, 0, 0))
    r_rate, r_c, r_n = tfidf_recall.get("relation", (0.0, 0, 0))
    report.append(f"| DART | TF-IDF | {e_rate:.4f} | {r_rate:.4f} | {e_rate-r_rate:+.4f} |")
    report.append(
        "\n**예상과 반대 방향.** relation 이름이 더 자연어에 가까운 DART에서 격차가 "
        "좁혀질 것으로 예상했지만, 실제로는 BERT·TF-IDF 모두 DART에서 격차가 훨씬 "
        "크게 벌어졌다(BERT: 1.6%p -> 28.7%p, TF-IDF: 18.8%p -> 31.4%p 안팎). "
        "**중요한 교란 요인**: DART WikiSQL/WikiTableText 서브셋은 WebNLG보다 표본이 "
        "훨씬 작고(train 4,968행 vs 16,023행, 약 31%) relation 어휘는 훨씬 길게 "
        "꼬리를 문다(고유 relation 1,066개 vs 346개, 즉 relation당 평균 예시 수가 "
        "WebNLG는 약 22개인데 DART는 약 2.2개뿈이다). relation 판정은 entity 판정보다 "
        "더 많은 문맥적 학습이 필요한 과제일 수 있는데, 학습 데이터와 relation당 "
        "반복 노출이 둘 다 크게 줄어든 상태라 '자연어성 효과'와 '데이터 희소성 효과'가 "
        "뒤섞여 있다. 이 둘을 분리하려면 WebNLG train을 DART와 같은 규모로 서브샘플링해서 "
        "같은 실험을 반복하는 ablation이 필요하다(자연어성은 그대로 둔 채 표본 크기만 "
        "맞추는 대조군).",
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
    report.append("\n## source(=category)별 오분류 분포 (참고)\n")
    report.append(f"- 전체 source 분포: {dict(cat_counter_all.most_common(10))}")
    report.append(f"- 오분류(noisy->grounded) source 분포: {dict(cat_counter_fn.most_common(10))}")
    report.append(f"- BERT entity corruption 오분류 {n_fn_entity}건 / relation corruption 오분류 {n_fn_relation}건")

    out_path = ERRORS_DIR / "dart_error_analysis_report.md"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"[18] 상세 리포트 저장 -> {out_path}")


if __name__ == "__main__":
    main()
