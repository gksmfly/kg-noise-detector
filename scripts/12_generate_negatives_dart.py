"""
Step 12 (도메인 일반화 탐침): DART WikiSQL/WikiTableText positive 쌍에 Step 2와
정확히 같은 corruption 방법론을 적용해 entity_corruption / relation_corruption
negative를 생성한다.

Step 2와 로직을 공유한다(`kg_noise.negatives.generate_negatives`) — 도메인만
다를 뿐 "표면적으로 트리플 문자열 한 슬롯만 바꾸고 나머지는 그대로 둔다"는
corruption 정의가 같아야 entity/relation 탐지 격차를 두 도메인 사이에서
공정하게 비교할 수 있다.
"""
from collections import Counter

from kg_noise.io_utils import load_jsonl, save_jsonl
from kg_noise.negatives import generate_negatives
from kg_noise.paths import PROCESSED_DIR, RAW_DIR

SRC = RAW_DIR / "dart_wikitable_pairs.jsonl"
OUT = PROCESSED_DIR / "dart_wikitable_full_dataset.jsonl"
SEED = 42


def main():
    rows = load_jsonl(SRC)
    print(f"[12] 원본 positive: {len(rows)}건")

    subject_pool = sorted(set(r["subject"] for r in rows))
    object_pool = sorted(set(r["object"] for r in rows))
    relation_pool = sorted(set(r["relation"] for r in rows))
    print(f"[12] 치환 풀: subject {len(subject_pool)} / object {len(object_pool)} / "
          f"relation {len(relation_pool)}")

    full = generate_negatives(rows, SEED)
    n_by_type = Counter(r["corruption_type"] for r in full)
    print(f"[12] 생성: positive {n_by_type['none']}건 + entity_corruption {n_by_type['entity']}건 + "
          f"relation_corruption {n_by_type['relation']}건 = {len(full)}건")

    save_jsonl(OUT, full)
    print(f"[12] 저장 -> {OUT}")


if __name__ == "__main__":
    main()
