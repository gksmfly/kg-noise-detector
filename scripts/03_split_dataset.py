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

from sklearn.model_selection import train_test_split

from kg_noise.io_utils import load_jsonl, save_jsonl
from kg_noise.paths import PROCESSED_DIR, SPLITS_DIR

SRC = PROCESSED_DIR / "full_dataset.jsonl"
OUT_DIR = SPLITS_DIR

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SEED = 42


def main():
    rows = load_jsonl(SRC)

    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)

    pair_ids = list(by_pair.keys())
    categories = [by_pair[pid][0]["category"] for pid in pair_ids]

    n_per_pair = Counter(len(v) for v in by_pair.values())
    print(f"[3] pair_id당 행 수 분포(3=positive+entity+relation 모두 살아남음): {dict(n_per_pair)}")
    print(f"[3] 전체 pair_id 수: {len(pair_ids)}, 전체 행 수: {len(rows)}")

    # category 중 표본이 너무 적어 stratify가 실패하는 경우를 대비해, 희귀 카테고리는
    # stratify 없이 처리되도록 최소 표본 수를 확인한다.
    cat_counts = Counter(categories)
    rare = {c for c, n in cat_counts.items() if n < 4}  # 70/30, 15/15 두 단계 stratify를 버티려면 최소 4개 필요
    if rare:
        print(f"[3] stratify 표본이 부족한(4건 미만) category {len(rare)}개는 stratify 없이 처리: {rare}")
        strat = None
    else:
        strat = categories

    train_ids, temp_ids, train_cat, temp_cat = train_test_split(
        pair_ids, categories, test_size=(VAL_RATIO + TEST_RATIO),
        stratify=strat, random_state=SEED,
    )
    val_ratio_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    temp_cat_counts = Counter(temp_cat)
    temp_strat = temp_cat if strat is not None and min(temp_cat_counts.values()) >= 2 else None
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_cat, test_size=(1 - val_ratio_of_temp),
        stratify=temp_strat, random_state=SEED,
    )

    splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    for name, ids in splits.items():
        split_rows = [r for pid in ids for r in by_pair[pid]]
        out_path = OUT_DIR / f"{name}.jsonl"
        save_jsonl(out_path, split_rows)

        corr_counter = Counter(r["corruption_type"] for r in split_rows)
        print(f"[3] {name}: pair {len(ids)}개, 행 {len(split_rows)}개, "
              f"corruption_type {dict(corr_counter)}")
        print(f"[3]   -> {out_path}")

    # leakage 방어적 검증: 같은 pair_id가 두 split에 동시에 들어가지 않았는지 확인
    all_ids_seen = set()
    overlap = set()
    for name, ids in splits.items():
        s = set(ids)
        overlap |= (all_ids_seen & s)
        all_ids_seen |= s
    print(f"[3] split 간 pair_id 중복(0이어야 정상): {len(overlap)}")


if __name__ == "__main__":
    main()
