"""
Step 14 (도메인 일반화 탐침): Step 4와 동일하게, DART WikiSQL/WikiTableText train
split에서 실제 토큰 길이 분포를 보고 이 도메인 전용 `max_length`를 정한다.

WebNLG의 max_length(48)를 그대로 재사용하지 않는 이유: 두 도메인을 apples-to-apples로
비교하려면 "각 도메인에서 그 도메인에 맞게 최적화된 모델끼리" 비교해야 한다. 다른
도메인의 길이 분포에 맞춰진 max_length를 그대로 쓰면 truncation 정도가 달라져서
recall 격차가 진짜 도메인 차이가 아니라 truncation 차이 때문일 수 있다.
"""
import json

import numpy as np
from transformers import BertTokenizerFast

from kg_noise.constants import MODEL_NAME
from kg_noise.io_utils import load_jsonl
from kg_noise.paths import DART_MAX_LENGTH_RECOMMENDATION_PATH, SPLITS_DIR

TRAIN = SPLITS_DIR / "dart_train.jsonl"


def main():
    rows = load_jsonl(TRAIN)

    tok = BertTokenizerFast.from_pretrained(MODEL_NAME)

    lengths = []
    for r in rows:
        enc = tok(r["triple_text"], r["sentence"], truncation=False)
        lengths.append(len(enc["input_ids"]))

    lengths = np.array(lengths)
    print(f"[14] DART train {len(rows)}건 (triple_text, sentence) 페어 토큰 길이 분포")
    for p in [50, 75, 90, 95, 97, 99, 100]:
        print(f"    p{p}: {np.percentile(lengths, p):.1f}")
    print(f"    max: {lengths.max()}  mean: {lengths.mean():.1f}")

    p95 = int(np.percentile(lengths, 95))
    recommended = ((p95 + 7) // 8) * 8  # 8의 배수로 올림 (패딩 효율)
    print(f"[14] 95th percentile: {p95} -> 8의 배수로 올림: {recommended}")

    out = DART_MAX_LENGTH_RECOMMENDATION_PATH
    with out.open("w", encoding="utf-8") as f:
        json.dump({
            "p95_raw": p95,
            "recommended_max_length": recommended,
            "n_train": len(rows),
        }, f, ensure_ascii=False, indent=2)
    print(f"[14] 저장 -> {out}")


if __name__ == "__main__":
    main()
