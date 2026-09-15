"""pair_id-level stratified train/val/test split, shared by every domain's
split script (originally only `scripts/03_split_dataset.py` for WebNLG; now
also `scripts/13_split_dataset_dart.py` for the DART WikiSQL/WikiTableText
probe).

Splits at the pair_id level (not the row level) so the positive/
entity_corruption/relation_corruption rows derived from the same original
pair always land in the same split — otherwise a corrupted row could leak
its "correct" counterpart across train/val/test.
"""
from collections import Counter

from sklearn.model_selection import train_test_split


def split_by_pair_id(
    rows: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_stratify_count: int = 4,
) -> tuple[dict[str, list[dict]], dict]:
    """Returns (splits, stats). splits maps "train"/"val"/"test" -> rows.
    stats carries the diagnostics the caller scripts print (pair/row counts,
    which categories were too rare to stratify on, leakage check)."""
    by_pair: dict[str, list[dict]] = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], []).append(r)

    pair_ids = list(by_pair.keys())
    categories = [by_pair[pid][0]["category"] for pid in pair_ids]

    n_per_pair = Counter(len(v) for v in by_pair.values())

    # category 중 표본이 너무 적어 stratify가 실패하는 경우를 대비해, 희귀 카테고리는
    # stratify 없이 처리되도록 최소 표본 수를 확인한다.
    cat_counts = Counter(categories)
    rare = {c for c, n in cat_counts.items() if n < min_stratify_count}
    strat = None if rare else categories

    train_ids, temp_ids, train_cat, temp_cat = train_test_split(
        pair_ids, categories, test_size=(val_ratio + test_ratio),
        stratify=strat, random_state=seed,
    )
    val_ratio_of_temp = val_ratio / (val_ratio + test_ratio)
    temp_cat_counts = Counter(temp_cat)
    temp_strat = temp_cat if strat is not None and min(temp_cat_counts.values()) >= 2 else None
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids, temp_cat, test_size=(1 - val_ratio_of_temp),
        stratify=temp_strat, random_state=seed,
    )

    id_splits = {"train": train_ids, "val": val_ids, "test": test_ids}
    splits = {name: [r for pid in ids for r in by_pair[pid]] for name, ids in id_splits.items()}

    # leakage 방어적 검증: 같은 pair_id가 두 split에 동시에 들어가지 않았는지 확인
    all_ids_seen: set = set()
    overlap: set = set()
    for ids in id_splits.values():
        s = set(ids)
        overlap |= (all_ids_seen & s)
        all_ids_seen |= s

    stats = {
        "n_per_pair": dict(n_per_pair),
        "n_pairs": len(pair_ids),
        "n_rows": len(rows),
        "rare_categories": rare,
        "id_splits": id_splits,
        "overlap_count": len(overlap),
    }
    return splits, stats
