"""
Step 10 (부록 2, Step 9의 truncation 교란 요인 분리): 같은 30건 NYT-FB 샘플을
max_length=128(30건 전부 안 잘림, 최대 길이 102토큰)로 재추론해서, Step 9에서 본
"거의 전부 noisy로 예측됨"이라는 관찰이 (a) truncation 때문인지 (b) 표기/문체 이동
자체 때문인지 분리한다.

주의 (리포트에도 그대로 남긴다):
  - 이 모델은 max_length=48로 padding="max_length" 고정해서 학습됐다. 즉 학습 중
    모든 배치가 정확히 48 토큰이었고, position 48~127에 해당하는 위치 임베딩은
    fine-tuning 과정에서 단 한 번도 gradient를 받지 않았다 (사전학습 BERT가 가진
    값 그대로 남아있음). 그래서 max_length=128 추론이 "성능을 개선"한다는 보장이
    전혀 없다 — 오히려 한 번도 학습 중에 보지 않은 위치의 표현이라 더 불안정할
    수 있다. 이건 "truncation을 없앴을 때 결과가 어떻게 달라지는지"를 보는
    진단용 실험이지, "성능이 좋아지는지" 확인하는 실험이 아니다.
  - 여전히 30건 규모의 정성적 탐침이다. 비율은 참고용일 뿐 accuracy가 아니다
    (NYT-FB에는 정답 라벨이 없다는 사실은 Step 9와 동일).

Step 9와 정확히 같은 30건을 얻기 위해 동일한 SEED로 동일한 셔플을 재현한다.
"""
import json
import random

from datasets import load_dataset

from kg_noise.constants import LABEL_GROUNDED
from kg_noise.inference import load_classifier, predict_pair, raw_token_length
from kg_noise.paths import ERRORS_DIR, MODEL_BEST_DIR

STEP9_JSONL = ERRORS_DIR / "nyt_case_study.jsonl"
OUT_JSONL = ERRORS_DIR / "nyt_notrunc_comparison.jsonl"
OUT_REPORT = ERRORS_DIR / "nyt_notrunc_comparison_report.md"

N_SAMPLES = 30
SEED = 42
MAX_LENGTH_TRAINED = 48   # Step 9와 동일 (학습 시 max_length)
MAX_LENGTH_NOTRUNC = 128  # 이번 30건의 최대 실제 길이(102)보다 여유 있게


def natural_relation(rtext: str) -> str:
    return rtext.strip("/").split("/")[-1].replace("_", " ")


def main():
    rng = random.Random(SEED)
    print("[10] NYT-FB (nyt10) 로드 중... (Step 9와 동일한 셔플 재현)")
    ds = load_dataset("xiaobendanyn/nyt10")["test"]
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    picked = indices[:N_SAMPLES]

    # Step 9 결과를 nyt_index 기준으로 불러와 48-length 예측을 재사용 (재계산 대신 재사용해도
    # 결정론적으로 같은 값이 나오지만, 재현성 검증 차원에서 재계산도 같이 해서 비교한다).
    step9_by_idx = {}
    if STEP9_JSONL.exists():
        with STEP9_JSONL.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                step9_by_idx[r["nyt_index"]] = r

    model, tokenizer, device = load_classifier(MODEL_BEST_DIR)

    results = []
    for idx in picked:
        ex = ds[idx]
        rel = ex["relations"][0]
        subject, obj, rtext = rel["em1"], rel["em2"], rel["rtext"]
        sentence = ex["sentext"]

        orig_triple = f"{subject} | {rtext} | {obj}"
        nat_rel = natural_relation(rtext)
        nat_triple = f"{subject} | {nat_rel} | {obj}"

        raw_len = raw_token_length(tokenizer, orig_triple, sentence)
        o48_pred, o48_prob = predict_pair(model, tokenizer, device, orig_triple, sentence, MAX_LENGTH_TRAINED)
        n48_pred, n48_prob = predict_pair(model, tokenizer, device, nat_triple, sentence, MAX_LENGTH_TRAINED)
        o128_pred, o128_prob = predict_pair(model, tokenizer, device, orig_triple, sentence, MAX_LENGTH_NOTRUNC)
        n128_pred, n128_prob = predict_pair(model, tokenizer, device, nat_triple, sentence, MAX_LENGTH_NOTRUNC)

        was_truncated_at_48 = raw_len > MAX_LENGTH_TRAINED
        results.append({
            "nyt_index": idx,
            "sentence": sentence,
            "orig_triple_text": orig_triple,
            "nat_triple_text": nat_triple,
            "raw_token_len": raw_len,
            "was_truncated_at_48": was_truncated_at_48,
            "orig_48": {"pred": o48_pred, "prob_grounded": o48_prob},
            "nat_48": {"pred": n48_pred, "prob_grounded": n48_prob},
            "orig_128": {"pred": o128_pred, "prob_grounded": o128_prob},
            "nat_128": {"pred": n128_pred, "prob_grounded": n128_prob},
        })

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[10] 원본 결과 저장 -> {OUT_JSONL}")

    def grounded_rate(cell: str) -> tuple[int, int]:
        c = sum(1 for r in results if r[cell]["pred"] == LABEL_GROUNDED)
        return c, len(results)

    cells = {
        ("Freebase 표기", "max_length=48 (학습 시)"): "orig_48",
        ("Freebase 표기", "max_length=128 (재추론)"): "orig_128",
        ("자연어 변환 표기", "max_length=48 (학습 시)"): "nat_48",
        ("자연어 변환 표기", "max_length=128 (재추론)"): "nat_128",
    }
    print("[10] === 2x2 grounded 예측 건수 (참고용, accuracy 아님) ===")
    grid = {}
    for (notation, length), key in cells.items():
        c, n = grounded_rate(key)
        grid[(notation, length)] = (c, n)
        print(f"[10] {notation:12s} x {length:24s}: grounded {c}/{n}")

    n_trunc = sum(1 for r in results if r["was_truncated_at_48"])
    print(f"[10] 48-length에서 실제로 잘렸던 샘플: {n_trunc}/{len(results)}")

    # truncation 해제로 예측이 바뀐 건수 (표기별)
    flip_orig_by_length = sum(1 for r in results if r["orig_48"]["pred"] != r["orig_128"]["pred"])
    flip_nat_by_length = sum(1 for r in results if r["nat_48"]["pred"] != r["nat_128"]["pred"])
    print(f"[10] max_length을 48->128로 바꿨을 때 예측이 바뀐 건수: "
          f"Freebase표기 {flip_orig_by_length}건, 자연어표기 {flip_nat_by_length}건")

    report = [
        "# Step 10: truncation 교란 요인 분리 (Step 9 후속)\n",
        "**여전히 accuracy가 아니다.** NYT-FB에는 정답 라벨이 없다. 아래 비율은 "
        "\"grounded로 예측된 건수\"일 뿐 \"맞게 예측한 건수\"가 아니다.\n",
        "**모델은 max_length=48로만 fine-tuning됐다.** 학습 중 모든 배치가 정확히 48 "
        "토큰으로 padding됐기 때문에, position 48~127에 대응하는 위치 임베딩은 "
        "fine-tuning 동안 한 번도 gradient를 받지 않았고 사전학습 BERT의 값 그대로다. "
        "따라서 max_length=128 추론이 \"더 정확해진다\"는 보장은 전혀 없다 — 이 실험은 "
        "성능 개선을 확인하는 게 아니라, truncation 여부가 \"거의 전부 noisy\"라는 "
        "관찰에 얼마나 기여했는지를 분리해서 보려는 진단용 재추론이다.\n",
        f"- 30건 중 max_length=48에서 실제로 잘렸던 샘플: {n_trunc}건",
        f"- max_length을 48->128로 바꿨을 때 예측이 바뀐 건수: "
        f"Freebase표기 {flip_orig_by_length}건 / 자연어표기 {flip_nat_by_length}건\n",
        "## 2x2 비교표: grounded로 예측된 건수 (30건 중)\n",
        "| | max_length=48 (학습 시, 원래 Step 9) | max_length=128 (재추론, truncation 거의 없음) |",
        "|---|---|---|",
        f"| Freebase 표기 (`/people/person/nationality`) | "
        f"{grid[('Freebase 표기', 'max_length=48 (학습 시)')][0]}/{grid[('Freebase 표기', 'max_length=48 (학습 시)')][1]} | "
        f"{grid[('Freebase 표기', 'max_length=128 (재추론)')][0]}/{grid[('Freebase 표기', 'max_length=128 (재추론)')][1]} |",
        f"| 자연어 변환 표기 (`nationality`) | "
        f"{grid[('자연어 변환 표기', 'max_length=48 (학습 시)')][0]}/{grid[('자연어 변환 표기', 'max_length=48 (학습 시)')][1]} | "
        f"{grid[('자연어 변환 표기', 'max_length=128 (재추론)')][0]}/{grid[('자연어 변환 표기', 'max_length=128 (재추론)')][1]} |",
        "",
        "## 해석 가이드",
        "",
        "- **48->128에서 grounded 건수가 크게 늘었다면**: truncation이 \"거의 전부 "
        "noisy\" 관찰에 상당 부분 기여했다는 뜻 — Step 9의 결과를 \"모델이 도메인 "
        "이동에 완전히 무너졌다\"고 해석하면 과장이 된다.",
        "- **128에서도 여전히 대부분 noisy라면**: truncation과 무관하게 표기/문체 "
        "이동 자체가 근본 원인이라는 더 강한 근거가 된다 — 이 경우 Step 9의 프레이밍이 "
        "정당화된다.",
        "- 어느 쪽이든, 위에서 계산한 \"48->128 예측 변화 건수\"가 실질적 답이다. "
        "이 숫자가 0에 가까우면 truncation은 주된 원인이 아니고, 크면 truncation이 "
        "상당 부분을 설명한다.",
        "",
        "## founders 사례 재확인 (Step 9의 유일한 grounded 사례)\n",
    ]

    founders_case = next(
        (r for r in results if "founders" in r["orig_triple_text"] and "endemol" in r["orig_triple_text"].lower()),
        None,
    )
    if founders_case:
        report.append(
            f"- `{founders_case['orig_triple_text']}` (원문 길이 {founders_case['raw_token_len']}토큰, "
            f"{'48에서 잘림' if founders_case['was_truncated_at_48'] else '48에서도 안 잘림'})"
        )
        report.append(
            f"  - max_length=48: Freebase P(grounded)={founders_case['orig_48']['prob_grounded']:.3f}, "
            f"자연어 P={founders_case['nat_48']['prob_grounded']:.3f}"
        )
        report.append(
            f"  - max_length=128: Freebase P(grounded)={founders_case['orig_128']['prob_grounded']:.3f}, "
            f"자연어 P={founders_case['nat_128']['prob_grounded']:.3f}"
        )
        report.append(
            "  - relation 단어(\"founders\")가 문장에 그대로 등장하는 이 사례가 128에서도 "
            "여전히 grounded로 남는지가, \"관계명-문장 어휘 중첩\" 패턴이 truncation과 "
            "무관하게 진짜인지 보여주는 핵심 체크포인트다."
        )

    report.append("\n## 30건 상세 (원문 토큰 길이·잘림 여부 포함)\n")
    report.append("| nyt_index | raw_len | 48서 잘림? | sentence | Freebase 48->128 | 자연어 48->128 |")
    report.append("|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda r: -r["raw_token_len"]):
        o = f"{r['orig_48']['prob_grounded']:.3f}->{r['orig_128']['prob_grounded']:.3f}"
        n = f"{r['nat_48']['prob_grounded']:.3f}->{r['nat_128']['prob_grounded']:.3f}"
        report.append(
            f"| {r['nyt_index']} | {r['raw_token_len']} | "
            f"{'YES' if r['was_truncated_at_48'] else 'no'} | {r['sentence'][:80]} | {o} | {n} |"
        )

    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"[10] 리포트 저장 -> {OUT_REPORT}")


if __name__ == "__main__":
    main()
