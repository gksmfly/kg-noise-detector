"""
Step 2: Negative(노이즈) 데이터 생성.

원본 (트리플, 문장) 쌍마다 negative를 **두 종류 모두** 하나씩 만든다:
  - entity_corruption: 트리플의 subject 또는 object(50/50 무작위 선택)를
    전체 엔티티 풀에서 다른 값으로 치환. 문장은 그대로 둔다.
  - relation_corruption: 트리플의 relation을 전체 relation 풀에서 다른 값으로 치환.
    문장은 그대로 둔다.

최종 positive : entity_corruption : relation_corruption = 1 : 1 : 1
(전체 데이터 = 원본 샘플 수 x 3).

문장을 건드리지 않고 트리플만 깨뜨리는 이유: "이 문장이 이 트리플을 실제로
서술하는가"를 판별하는 과제이므로, negative의 정의가 "트리플이 문장과 안 맞음"이어야
한다. 두 corruption 모두 표면적으로는 트리플 문자열 안의 토큰 하나만 바뀌고
나머지 입력(문장 전체 + 트리플의 나머지 두 슬롯)은 동일하게 유지된다는 점에서
대칭적으로 설계되어 있다 — 이래야 "어느 쪽이 더 탐지하기 어려운가"라는 연구
질문이 공정한 비교가 된다.
"""
from collections import Counter

from kg_noise.io_utils import load_jsonl, save_jsonl
from kg_noise.negatives import generate_negatives
from kg_noise.paths import PROCESSED_DIR, RAW_DIR

SRC = RAW_DIR / "webnlg_pairs.jsonl"
OUT = PROCESSED_DIR / "full_dataset.jsonl"
SEED = 42


def main():
    rows = load_jsonl(SRC)
    print(f"[2] 원본 positive: {len(rows)}건")

    subject_pool = sorted(set(r["subject"] for r in rows))
    object_pool = sorted(set(r["object"] for r in rows))
    relation_pool = sorted(set(r["relation"] for r in rows))
    print(f"[2] 치환 풀: subject {len(subject_pool)} / object {len(object_pool)} / "
          f"relation {len(relation_pool)}")

    full = generate_negatives(rows, SEED)
    n_by_type = Counter(r["corruption_type"] for r in full)
    print(f"[2] 생성: positive {n_by_type['none']}건 + entity_corruption {n_by_type['entity']}건 + "
          f"relation_corruption {n_by_type['relation']}건 = {len(full)}건")

    save_jsonl(OUT, full)
    print(f"[2] 저장 -> {OUT}")


if __name__ == "__main__":
    main()
