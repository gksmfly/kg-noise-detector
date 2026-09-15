"""
Step 3: 데이터 분할.

분할 단위는 pair_id다 (같은 원본 트리플-문장 쌍에서 파생된 positive/entity_corruption/
relation_corruption 3행이 서로 다른 split에 나뉘어 들어가면 안 된다 - leakage 방지).
각 pair_id는 정확히 3행을 가지므로, pair_id 단위로 먼저 train/val/test를 나누고
나서 그 안에 속한 모든 행을 그대로 가져온다.

stratify 기준은 category. sklearn train_test_split을 두 번 적용해 70/15/15를 만든다
(먼저 70/30, 그다음 30을 15/15로).
"""
from collections import Counter

from kg_noise.io_utils import load_jsonl, save_jsonl
from kg_noise.paths import PROCESSED_DIR, SPLITS_DIR
from kg_noise.splitting import split_by_pair_id

SRC = PROCESSED_DIR / "full_dataset.jsonl"
OUT_DIR = SPLITS_DIR

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SEED = 42


def main():
    rows = load_jsonl(SRC)

    splits, stats = split_by_pair_id(rows, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)

    print(f"[3] pair_id당 행 수 분포(3=positive+entity+relation 모두 살아남음): {stats['n_per_pair']}")
    print(f"[3] 전체 pair_id 수: {stats['n_pairs']}, 전체 행 수: {stats['n_rows']}")
    if stats["rare_categories"]:
        print(f"[3] stratify 표본이 부족한(4건 미만) category {len(stats['rare_categories'])}개는 "
              f"stratify 없이 처리: {stats['rare_categories']}")

    for name, split_rows in splits.items():
        out_path = OUT_DIR / f"{name}.jsonl"
        save_jsonl(out_path, split_rows)

        corr_counter = Counter(r["corruption_type"] for r in split_rows)
        print(f"[3] {name}: pair {len(stats['id_splits'][name])}개, 행 {len(split_rows)}개, "
              f"corruption_type {dict(corr_counter)}")
        print(f"[3]   -> {out_path}")

    print(f"[3] split 간 pair_id 중복(0이어야 정상): {stats['overlap_count']}")


if __name__ == "__main__":
    main()
