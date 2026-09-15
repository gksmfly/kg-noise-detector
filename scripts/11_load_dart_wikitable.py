"""
Step 11 (도메인 일반화 탐침, 본실험과 apples-to-apples): DART에서 WikiSQL/
WikiTableText 계열 (트리플, 문장) 쌍을 로드한다.

왜 이 서브셋인가 — WebNLG와의 핵심 차이점:
WebNLG의 relation은 DBpedia/Freebase 스타일 카멜케이스 식별자다(예: `cityServed`,
`foundedBy`). 이런 식별자는 문장 표면에 거의 그대로 등장하지 않는다 — 이게 본실험에서
relation_corruption이 entity_corruption보다 탐지하기 어려웠던 핵심 메커니즘이었다
(TF-IDF 등 표면 어휘 기반 방법이 relation corruption에 거의 무방비였음, Step 8 참고).

DART는 여러 소스를 묶은 오픈도메인 record-to-text 데이터셋인데, 그중 WikiSQL과
WikiTableText는 위키피디아 "표"에서 파생됐다. 이 표들의 relation은 표의 컬럼
헤더다(예: `COLLEGE`, `CITY`, `FEET`, `TOTAL_NUMBER_OF_STUDENTS`) — 대문자·언더스코어
표기지만 사람이 읽을 수 있는 자연어 단어에 훨씬 가깝다. 즉 "relation 식별자가 얼마나
자연어에 가까운가"라는 축을 인위적으로 바꾼 대조군이 된다: 같은 corruption 방법론을
그대로 적용했을 때, entity/relation 탐지 격차가 좁혀지는지(가설의 메커니즘이 맞다는
증거) 아니면 그대로인지(다른 요인이 있다는 증거)를 검증한다.

데이터 처리:
- `GEM/dart`도 `datasets>=4`에서 로딩 스크립트가 막혀서(01과 동일한 문제),
  `refs/convert/parquet` 리비전을 직접 읽는다.
- DART의 `target_sources` 필드로 소스를 구분한다. `WikiSQL_decl_sents`,
  `WikiTableQuestions_lily`, `WikiTableQuestions_mturk` 세 소스만 쓴다(둘 다
  위키 표 파생, 자연어에 가까운 relation 이름). e2e(레스토랑 도메인, 속성이 손에 꼽힘)와
  webnlg(이미 본실험에 있음)는 제외.
- `tripleset`이 리스트라 여러 트리플을 가진 샘플이 많다 — 01과 동일한 이유로
  단일 트리플 샘플만 쓴다(len(tripleset) == 1).
- 표 메타데이터 마커(`[TABLECONTEXT]`, `[TITLE]`)가 트리플 자리에 섞여 들어간
  샘플이 있어(DART가 표 제목/컨텍스트를 트리플처럼 직렬화한 경우), 이런 샘플은
  스킵한다.
- `category` 필드가 없으므로(WebNLG는 있었음) `target_sources`(3가지 값)를
  대신 stratify 기준으로 쓴다.
"""
from collections import Counter

from datasets import load_dataset

from kg_noise.io_utils import save_jsonl
from kg_noise.paths import RAW_DIR

OUT = RAW_DIR / "dart_wikitable_pairs.jsonl"

PARQUET_URL = (
    "https://huggingface.co/datasets/GEM/dart/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)

TARGET_SOURCES = {"WikiSQL_decl_sents", "WikiTableQuestions_lily", "WikiTableQuestions_mturk"}
TABLE_MARKERS = {"[TABLECONTEXT]", "[TITLE]"}


def main():
    print("[11] GEM/dart (train) parquet 로드 중...")
    ds = load_dataset("parquet", data_files={"train": PARQUET_URL})["train"]
    print(f"[11] 원본 로드: {len(ds)}건")

    rows = []
    skipped_other_source = 0
    skipped_multi = 0
    skipped_table_marker = 0
    seen_triple_sentence = set()
    for ex in ds:
        srcs = ex["target_sources"]
        if len(srcs) != 1 or srcs[0] not in TARGET_SOURCES:
            skipped_other_source += 1
            continue
        triples = ex["tripleset"]
        if len(triples) != 1:
            skipped_multi += 1
            continue
        subject, relation, obj = triples[0]
        if any(v in TABLE_MARKERS for v in (subject, relation, obj)):
            skipped_table_marker += 1
            continue
        sentence = (ex["target"] or "").strip()
        subject, relation, obj = subject.strip(), relation.strip(), obj.strip()
        if not sentence or not subject or not relation or not obj:
            continue
        triple_text = f"{subject} | {relation} | {obj}"
        key = (triple_text, sentence)
        if key in seen_triple_sentence:
            continue
        seen_triple_sentence.add(key)
        rows.append({
            "pair_id": f"dart:{len(rows)}",
            "subject": subject,
            "relation": relation,
            "object": obj,
            "triple_text": triple_text,
            "sentence": sentence,
            "category": srcs[0],
        })

    print(f"[11] 단일 트리플 WikiSQL/WikiTableText 샘플: {len(rows)}건 "
          f"(다른 소스 제외 {skipped_other_source}건, 다중 트리플 제외 {skipped_multi}건, "
          f"표 메타데이터 마커 제외 {skipped_table_marker}건)")

    cat_counter = Counter(r["category"] for r in rows)
    print(f"[11] source(=category) 분포: {dict(cat_counter)}")

    n_subjects = len(set(r["subject"] for r in rows))
    n_relations = len(set(r["relation"] for r in rows))
    n_objects = len(set(r["object"] for r in rows))
    print(f"[11] 고유 subject {n_subjects}개, relation {n_relations}개, object {n_objects}개")

    save_jsonl(OUT, rows)
    print(f"[11] 저장 -> {OUT}")


if __name__ == "__main__":
    main()
