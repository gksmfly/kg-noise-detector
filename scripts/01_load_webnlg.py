"""
Step 1: WebNLG 데이터셋에서 (트리플, 문장) 쌍을 로드한다.

- 원천: GEM/web_nlg (HuggingFace, en, train split).
  `load_dataset("GEM/web_nlg", "en")`은 datasets>=4 에서
  "Dataset scripts are no longer supported"로 막힌다 (레거시 로딩 스크립트 방식이라
  최신 datasets 라이브러리가 거부함, 실측 확인). 대신 HuggingFace가 자동 변환해둔
  parquet 파일(refs/convert/parquet 브랜치)을 datasets의 "parquet" 빌더로 직접 읽는다.
  필드 구조는 원본과 동일하다 (실제 샘플로 확인):
    input: 트리플 문자열의 리스트. 각 원소는 "Subject | relation | Object" 형식.
           예: ['Aarhus_Airport | cityServed | "Aarhus, Denmark"']
           멀티 트리플 샘플은 리스트 길이가 2 이상 (예: 3개 트리플을 하나의 문장이 서술).
    target: 그 트리플(들)을 서술하는 문장(사람이 작성).
    category: 도메인 카테고리 (Airport, Building, ...).
- 트리플이 여러 개인 샘플은 "문장의 어느 부분이 어느 트리플에 대응하는지"가
  모호해지므로 스코프 밖으로 제외한다 (연구 설계서에 명시된 스코프 제한).
  len(input) == 1인 샘플만 사용한다.
- 트리플 문자열은 "Subject | relation | Object" 형식이라 " | " 기준으로 분리해서
  파싱한다.
"""
from collections import Counter

from datasets import load_dataset

from kg_noise.io_utils import save_jsonl
from kg_noise.paths import RAW_DIR

OUT = RAW_DIR / "webnlg_pairs.jsonl"

PARQUET_URL = (
    "https://huggingface.co/datasets/GEM/web_nlg/resolve/"
    "refs%2Fconvert%2Fparquet/en/train/0000.parquet"
)


def parse_triple(triple_str: str) -> tuple[str, str, str] | None:
    parts = [p.strip() for p in triple_str.split(" | ")]
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def main():
    print("[1] GEM/web_nlg (en, train) parquet 로드 중...")
    ds = load_dataset("parquet", data_files={"train": PARQUET_URL})["train"]
    print(f"[1] 원본 로드: {len(ds)}건")

    rows = []
    skipped_multi = 0
    skipped_unparsable = 0
    seen_triple_sentence = set()  # (triple_text, sentence) 중복 제거
    for ex in ds:
        triples = ex["input"]
        if len(triples) != 1:
            skipped_multi += 1
            continue
        parsed = parse_triple(triples[0])
        if parsed is None:
            skipped_unparsable += 1
            continue
        subject, relation, obj = parsed
        sentence = (ex["target"] or "").strip()
        if not sentence or not subject or not relation or not obj:
            continue
        triple_text = f"{subject} | {relation} | {obj}"
        key = (triple_text, sentence)
        if key in seen_triple_sentence:
            continue
        seen_triple_sentence.add(key)
        rows.append({
            "pair_id": f"webnlg:{len(rows)}",
            "subject": subject,
            "relation": relation,
            "object": obj,
            "triple_text": triple_text,
            "sentence": sentence,
            "category": ex["category"],
        })

    print(f"[1] 단일 트리플 샘플: {len(rows)}건 "
          f"(다중 트리플 제외 {skipped_multi}건, 파싱 실패 제외 {skipped_unparsable}건)")

    cat_counter = Counter(r["category"] for r in rows)
    print(f"[1] category 분포 (상위 10개): {cat_counter.most_common(10)}")

    n_subjects = len(set(r["subject"] for r in rows))
    n_relations = len(set(r["relation"] for r in rows))
    n_objects = len(set(r["object"] for r in rows))
    print(f"[1] 고유 subject {n_subjects}개, relation {n_relations}개, object {n_objects}개")

    save_jsonl(OUT, rows)
    print(f"[1] 저장 -> {OUT}")


if __name__ == "__main__":
    main()
