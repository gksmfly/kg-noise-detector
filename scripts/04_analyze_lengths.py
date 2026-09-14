"""
Step 4: tokenizer(triple_text, sentence) 인코딩 전에 실제 길이 분포를 보고
max_length를 정한다. train split 기준 95th percentile로 정한다
(테스트 통계 유출을 피하기 위해 train만 사용).
"""
import json

import numpy as np
from transformers import BertTokenizerFast

from kg_noise.constants import MODEL_NAME
from kg_noise.io_utils import load_jsonl
from kg_noise.paths import SPLITS_DIR

TRAIN = SPLITS_DIR / "train.jsonl"


def main():
    rows = load_jsonl(TRAIN)

    tok = BertTokenizerFast.from_pretrained(MODEL_NAME)

    lengths = []
    for r in rows:
        enc = tok(r["triple_text"], r["sentence"], truncation=False)
        lengths.append(len(enc["input_ids"]))

    lengths = np.array(lengths)
    print(f"[4] train {len(rows)}건 (triple_text, sentence) 페어 토큰 길이 분포")
    for p in [50, 75, 90, 95, 97, 99, 100]:
        print(f"    p{p}: {np.percentile(lengths, p):.1f}")
    print(f"    max: {lengths.max()}  mean: {lengths.mean():.1f}")

    p95 = int(np.percentile(lengths, 95))
    recommended = ((p95 + 7) // 8) * 8  # 8의 배수로 올림 (패딩 효율)
    print(f"[4] 95th percentile: {p95} -> 8의 배수로 올림: {recommended}")

    out = SPLITS_DIR / "max_length_recommendation.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump({
            "p95_raw": p95,
            "recommended_max_length": recommended,
            "n_train": len(rows),
        }, f, ensure_ascii=False, indent=2)
    print(f"[4] 저장 -> {out}")


if __name__ == "__main__":
    main()
