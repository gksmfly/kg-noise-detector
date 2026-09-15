"""
Step 13 (도메인 일반화 탐침): DART WikiSQL/WikiTableText 데이터를 Step 3와 동일한
방식(pair_id 단위 70/15/15, stratify)으로 분할한다.

WebNLG는 `category`(Airport, Building, ...)로 stratify했지만, DART엔 그런 도메인
카테고리가 없다. 대신 `category` 필드에 이미 소스 이름(`WikiSQL_decl_sents`,
`WikiTableQuestions_lily`, `WikiTableQuestions_mturk`, Step 11 참고)을 넣어뒀으므로
그대로 stratify 기준으로 쓴다.
"""
from collections import Counter

from kg_noise.io_utils import load_jsonl, save_jsonl
from kg_noise.paths import PROCESSED_DIR, SPLITS_DIR
from kg_noise.splitting import split_by_pair_id

SRC = PROCESSED_DIR / "dart_wikitable_full_dataset.jsonl"
OUT_DIR = SPLITS_DIR

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SEED = 42


def main():
    rows = load_jsonl(SRC)

    splits, stats = split_by_pair_id(rows, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, SEED)

    print(f"[13] pair_id당 행 수 분포(3=positive+entity+relation 모두 살아남음): {stats['n_per_pair']}")
    print(f"[13] 전체 pair_id 수: {stats['n_pairs']}, 전체 행 수: {stats['n_rows']}")
    if stats["rare_categories"]:
        print(f"[13] stratify 표본이 부족한(4건 미만) source {len(stats['rare_categories'])}개는 "
              f"stratify 없이 처리: {stats['rare_categories']}")

    for name, split_rows in splits.items():
        out_path = OUT_DIR / f"dart_{name}.jsonl"
        save_jsonl(out_path, split_rows)

        corr_counter = Counter(r["corruption_type"] for r in split_rows)
        print(f"[13] {name}: pair {len(stats['id_splits'][name])}개, 행 {len(split_rows)}개, "
              f"corruption_type {dict(corr_counter)}")
        print(f"[13]   -> {out_path}")

    print(f"[13] split 간 pair_id 중복(0이어야 정상): {stats['overlap_count']}")


if __name__ == "__main__":
    main()
