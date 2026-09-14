"""
Step 9 (부록, 정성적 케이스 스터디): NYT-FB(distant supervision) 데이터에서
"문체 이동 + relation 표기 체계 이동"이 실제로 모델을 어떻게 망가뜨리는지 관찰한다.

주의 — 이건 "일반화 성능 측정"이 아니다:
NYT-FB(nyt10)는 distant supervision으로 자동 라벨링된 데이터라 "이 문장이 이 트리플을
실제로 나타내는가"에 대한 사람 검증 정답이 없다. 그래서 accuracy/F1 같은 숫자를 낼 수
없고, 내서도 안 된다 (정답 없는 데이터에 정확도를 계산하는 건 방법론적으로 틀렸다).
이 스크립트는 각 샘플에 대해 두 버전의 예측만 뽑아 나란히 보여준다:
  1) 원본 표기: "Alex_Salmond | /people/person/nationality | Scotland"
  2) 자연어 변환: relation 경로의 마지막 토큰만 취해 "Alex_Salmond | nationality | Scotland"
     (완벽한 매핑이 아니라 "이 정도 처리로도 예측이 달라지는지"를 보는 용도)
어떤 사례가 흥미로운지(표기 차이로 예측이 뒤집힌 케이스 등)는 사람이 결과 파일을 직접
보고 3~5개를 골라야 한다 — 이건 자동화할 수 없는 부분이라 여기서는 "뒤집혔는지" 여부만
플래그로 달아 사람이 빠르게 스캔할 수 있게 해둔다.

우리 BERT는 WebNLG로만 학습됐고 NYT10을 본 적이 없으므로 이 모든 예측은 zero-shot이다.
"""
import json
import random

from datasets import load_dataset

from kg_noise.constants import LABEL_GROUNDED
from kg_noise.inference import load_classifier, predict_pair
from kg_noise.paths import ERRORS_DIR, MODEL_BEST_DIR, SPLITS_DIR

OUT_JSONL = ERRORS_DIR / "nyt_case_study.jsonl"
OUT_REPORT = ERRORS_DIR / "nyt_case_study_report.md"

N_SAMPLES = 30
SEED = 42


def natural_relation(rtext: str) -> str:
    """"/people/person/nationality" -> "nationality" (경로 마지막 토큰, 밑줄->공백)."""
    last = rtext.strip("/").split("/")[-1]
    return last.replace("_", " ")


def main():
    rng = random.Random(SEED)

    print("[9] NYT-FB (nyt10) 로드 중...")
    ds = load_dataset("xiaobendanyn/nyt10")["test"]
    print(f"[9] 전체 {len(ds)}건 (전부 relation 보유 — NA 버킷 없음)")

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    picked = indices[:N_SAMPLES]

    print(f"[9] {MODEL_BEST_DIR}에서 WebNLG로만 학습된 BERT 로드 (NYT10은 처음 봄 — zero-shot)")
    model, tokenizer, device = load_classifier(MODEL_BEST_DIR)

    max_length_path = SPLITS_DIR / "max_length_recommendation.json"
    max_length = json.loads(max_length_path.read_text())["recommended_max_length"]
    print(f"[9] max_length={max_length} (Step 4에서 정한 값을 그대로 사용)")

    results = []
    for idx in picked:
        ex = ds[idx]
        rel = ex["relations"][0]  # 여러 개면 첫 번째만 사용 (연구 스코프: 트리플 1개 기준 유지)
        subject, obj, rtext = rel["em1"], rel["em2"], rel["rtext"]
        sentence = ex["sentext"]

        orig_triple = f"{subject} | {rtext} | {obj}"
        nat_rel = natural_relation(rtext)
        nat_triple = f"{subject} | {nat_rel} | {obj}"

        orig_pred, orig_prob = predict_pair(model, tokenizer, device, orig_triple, sentence, max_length)
        nat_pred, nat_prob = predict_pair(model, tokenizer, device, nat_triple, sentence, max_length)

        results.append({
            "nyt_index": idx,
            "sentence": sentence,
            "subject": subject, "object": obj, "relation_raw": rtext, "relation_natural": nat_rel,
            "orig_triple_text": orig_triple,
            "orig_pred": orig_pred, "orig_prob_grounded": orig_prob,
            "nat_triple_text": nat_triple,
            "nat_pred": nat_pred, "nat_prob_grounded": nat_prob,
            "flipped": orig_pred != nat_pred,
        })

    n_flipped = sum(1 for r in results if r["flipped"])
    n_orig_grounded = sum(1 for r in results if r["orig_pred"] == LABEL_GROUNDED)
    n_nat_grounded = sum(1 for r in results if r["nat_pred"] == LABEL_GROUNDED)
    print(f"[9] {len(results)}건 중 표기 방식에 따라 예측이 뒤집힌 사례: {n_flipped}건")
    print(f"[9] grounded로 예측된 비율: 원본표기 {n_orig_grounded}/{len(results)}, "
          f"자연어표기 {n_nat_grounded}/{len(results)}")
    print("[9] (참고용 집계일 뿐 accuracy 아님 — NYT-FB엔 정답 라벨이 없음)")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[9] 원본 결과 저장 -> {OUT_JSONL}")

    # 사람이 빠르게 훑어볼 수 있도록, 뒤집힌 사례를 위로 올린 마크다운 표 생성
    results_sorted = sorted(results, key=lambda r: (not r["flipped"], r["nyt_index"]))
    lines = [
        "# NYT-FB 정성적 케이스 스터디 (zero-shot, WebNLG로만 학습된 BERT)\n",
        "**주의**: 이 표는 accuracy가 아니다. NYT-FB(distant supervision)에는 "
        "\"이 문장이 이 트리플을 실제로 나타내는가\"에 대한 사람 검증 정답이 없다. "
        "아래는 원본 Freebase 표기(`/people/person/nationality`)와 자연어 변환 표기"
        "(`nationality`) 각각으로 예측했을 때 결과가 어떻게 달라지는지 보여주는 "
        "정성적 비교일 뿐이다. **표기 차이로 예측이 뒤집힌(flipped=True) 사례부터 "
        "3~5개를 직접 읽고 케이스 스터디로 고를 것.**\n",
        f"- 표본 {len(results)}건, 표기 방식에 따라 예측이 뒤집힌 사례 {n_flipped}건",
        f"- grounded 예측 비율: 원본표기 {n_orig_grounded}/{len(results)} vs "
        f"자연어표기 {n_nat_grounded}/{len(results)} (참고용 집계, 정답 없음)\n",
        "| flipped | sentence | subject \\| relation(raw) \\| object | 원본 pred (P=grounded) | "
        "subject \\| relation(natural) \\| object | 자연어 pred (P=grounded) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results_sorted:
        flip_mark = "**YES**" if r["flipped"] else "no"
        orig_label = "grounded" if r["orig_pred"] == LABEL_GROUNDED else "noisy"
        nat_label = "grounded" if r["nat_pred"] == LABEL_GROUNDED else "noisy"
        lines.append(
            f"| {flip_mark} | {r['sentence'][:100]} | `{r['orig_triple_text']}` | "
            f"{orig_label} ({r['orig_prob_grounded']:.3f}) | `{r['nat_triple_text']}` | "
            f"{nat_label} ({r['nat_prob_grounded']:.3f}) |"
        )

    lines.append("\n## 케이스 스터디 (직접 골라 채울 것)\n")
    lines.append("- [ ] 사례 1:")
    lines.append("- [ ] 사례 2:")
    lines.append("- [ ] 사례 3:")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"[9] 마크다운 표 저장 -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
