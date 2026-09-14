# KG-Noise-Lite

BERT 분류기는 KG-to-text 데이터에서 **entity 단위**와 **relation 단위** 라벨
노이즈를 동일하게 잘 탐지할까, 아니면 한쪽이 구조적으로 더 어려울까?

## 연구 질문

Distant-supervision 방식의 KG-to-text 데이터셋에는 흔히 두 종류의 라벨
노이즈가 섞여 있다: 트리플의 entity가 문장과 안 맞거나, relation이 안 맞는
경우다. 이 프로젝트는 `bert-base-cased`를 grounded/noisy 이진 분류기로
파인튜닝하고, 이 모델과 TF-IDF 코사인 유사도 베이스라인이 두 노이즈 유형을
동일하게 잘 탐지하는지 측정한다.

**가설:** entity corruption은 표면적 어휘 불일치로 나타나 탐지가 쉬울 것이다
(예: *서울 → 부산*). relation corruption은 어휘 중첩은 유지한 채 문장의
의미만 깨뜨리므로 구조적으로 더 어려울 것이다(예: *수도 → 최대도시*).

## 데이터

- **출처:** [WebNLG](https://huggingface.co/datasets/GEM/web_nlg)
  (`GEM/web_nlg`, 영어, train split)를 HuggingFace가 자동 변환해둔 parquet
  파일에서 로드한다 — `load_dataset("GEM/web_nlg", "en")`은 `datasets>=4`에서
  실패하므로("Dataset scripts are no longer supported"),
  [`scripts/01_load_webnlg.py`](scripts/01_load_webnlg.py)가
  `refs/convert/parquet` 리비전을 직접 읽는다. 필드 구조는 원본과 동일하다:
  `input` = 트리플 문자열(`"Subject | relation | Object"`), `target` = 참조
  문장, `category` = 도메인 카테고리.
- 단일 트리플 예시만 사용한다(35,426건 중 7,630건) — 여러 트리플을 서술하는
  문장은 "문장의 어느 부분이 어느 트리플에 대응하는지"가 모호해지므로
  스코프 밖으로 제외한다(아래 스코프 참고).
- **Positive:** 원본 (트리플, 문장) 쌍.
- **Negative**, 두 종류를 positive마다 하나씩 생성(positive :
  entity_corruption : relation_corruption = 1 : 1 : 1, 총 22,890행):
  - `entity_corruption` — 트리플의 subject 또는 object(50/50)를 코퍼스
    전체 entity 풀에서 다른 값으로 치환. 문장은 그대로 둔다.
  - `relation_corruption` — 트리플의 relation을 코퍼스 전체 relation
    풀(346개 고유 relation)에서 다른 값으로 치환. 문장은 그대로 둔다.

## 파이프라인

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 1 | [`scripts/01_load_webnlg.py`](scripts/01_load_webnlg.py) | WebNLG 로드, 단일 트리플만 필터링, `"S \| R \| O"` 파싱, 중복 제거. |
| 2 | [`scripts/02_generate_negatives.py`](scripts/02_generate_negatives.py) | positive마다 `entity_corruption`과 `relation_corruption` negative를 하나씩 생성. |
| 3 | [`scripts/03_split_dataset.py`](scripts/03_split_dataset.py) | `pair_id` 단위로 70/15/15 분할(같은 pair의 3행은 항상 같은 split에), `category`로 stratify. |
| 4 | [`scripts/04_analyze_lengths.py`](scripts/04_analyze_lengths.py) | train split에서 토큰화된 `(triple_text, sentence)` 길이를 측정해 `max_length`를 추천. |
| 5 | [`scripts/05_train_bert.py`](scripts/05_train_bert.py) | `bert-base-cased`를 이진 sequence-pair 분류기로 파인튜닝, validation F1(macro) 기준 early stopping. |
| 6 | [`scripts/06_tfidf_baseline.py`](scripts/06_tfidf_baseline.py) | TF-IDF + 코사인 유사도 베이스라인; threshold는 validation F1(macro) grid search로 선택. |
| 7 | [`scripts/07_evaluate_compare.py`](scripts/07_evaluate_compare.py) | 두 모델의 test 지표를 표 하나로 합친다. |
| 8 | [`scripts/08_error_analysis.py`](scripts/08_error_analysis.py) | **핵심 결과:** 두 모델 모두 `entity_corruption` vs `relation_corruption`별 recall 분해 + 정성적 오류 사례. |

## 환경 설정

이 프로젝트는 이 기기의 다른 프로젝트와 분리된 자체 venv를 쓴다. 로컬
드라이버에 맞는 CUDA 빌드의 torch가 필요하기 때문이다:

```bash
python3 -m venv .venv
.venv/bin/pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install transformers datasets scikit-learn numpy accelerate
.venv/bin/pip install -e . --no-deps
```

(`pip install torch`만 실행하면 최신 CUDA 빌드를 받아온다 — 작성 시점 기준
CUDA 13 — 그런데 이게 오래된 드라이버에서는 `torch.cuda.is_available()`이
조용히 실패한다. 드라이버가 실제로 지원하는 CUDA 빌드로 고정할 것;
`nvidia-smi`로 드라이버가 지원하는 최대 CUDA 버전을 확인할 수 있다.)

마지막 줄은 이 저장소 자체의 [`src/kg_noise/`](src/kg_noise) 패키지를
editable 모드로 설치한다(런타임 의존성은 바로 위에서 이미 설치했으므로
`--no-deps`) — 이 덕분에 모든 `scripts/NN_*.py` 파일이 동일한 경로/IO/지표
헬퍼를 중복 정의하는 대신 `from kg_noise import ...`로 가져다 쓸 수 있다.

## 디렉터리 구조

```
.
├── pyproject.toml          # editable 패키지 설치 설정 (src 레이아웃)
├── README.md
├── .gitignore
├── src/
│   └── kg_noise/            # 여러 스크립트가 공유하는 라이브러리 코드
│       ├── __init__.py
│       ├── paths.py          # ROOT 및 data/models 하위 경로 상수
│       ├── io_utils.py       # JSONL 읽기/쓰기 (load_jsonl, save_jsonl)
│       ├── constants.py      # 라벨 인코딩(LABEL_NOISY/LABEL_GROUNDED), 베이스 모델명
│       ├── metrics.py        # BERT·TF-IDF가 공통으로 쓰는 분류 지표
│       └── inference.py      # 파인튜닝된 분류기 로딩 + 추론 (스크립트 9, 10용)
├── scripts/                 # 번호가 매겨진 파이프라인 진입점 (01~10, 순서대로 실행)
│   ├── 01_load_webnlg.py
│   ├── 02_generate_negatives.py
│   ├── 03_split_dataset.py
│   ├── 04_analyze_lengths.py
│   ├── 05_train_bert.py
│   ├── 06_tfidf_baseline.py
│   ├── 07_evaluate_compare.py
│   ├── 08_error_analysis.py
│   ├── 09_nyt_case_study.py          # 부록 (아래 참고)
│   └── 10_nyt_notrunc_comparison.py  # 부록 (아래 참고)
├── data/                     # 파이프라인 입출력 (raw/processed/splits/errors)
└── models/                   # 파인튜닝 체크포인트 (.gitignore로 추적 제외, scripts/05로 재생성 가능)
```

각 `scripts/NN_*.py` 파일은 재사용 로직을 담는 곳이 아니라 `src/kg_noise/`를
불러와 순서대로 실행만 하는 얇은 진입점이다. 여러 스크립트에서 같은 코드가
필요해지면 그 로직은 `scripts/`가 아니라 `src/kg_noise/`에 추가한다.

## 사용법

```bash
.venv/bin/python scripts/01_load_webnlg.py
.venv/bin/python scripts/02_generate_negatives.py
.venv/bin/python scripts/03_split_dataset.py
.venv/bin/python scripts/04_analyze_lengths.py
.venv/bin/python scripts/05_train_bert.py
.venv/bin/python scripts/06_tfidf_baseline.py
.venv/bin/python scripts/07_evaluate_compare.py
.venv/bin/python scripts/08_error_analysis.py
```

기기에 GPU가 2개 이상 있으면, [`scripts/05_train_bert.py`](scripts/05_train_bert.py)는
torch를 import하기 전에 `CUDA_VISIBLE_DEVICES=0`을 고정한다 — 이게 없으면
HuggingFace `Trainer`가 보이는 모든 GPU에 모델을 `DataParallel`로 감싸려다
이 환경에서 NCCL 오류로 죽는다.

## 결과

Test set (3,435행: positive 1,145 / entity_corruption 1,145 /
relation_corruption 1,145):

| 지표 | BERT (fine-tuned) | TF-IDF baseline |
|---|---|---|
| Accuracy | 0.9875 | 0.5872 |
| Precision (macro) | 0.9850 | 0.5418 |
| Recall (macro) | 0.9869 | 0.5430 |
| F1 (macro) | 0.9859 | 0.5422 |
| Recall (noisy) | 0.9886 | 0.6755 |
| Precision (noisy) | 0.9925 | 0.6962 |
| F1 (noisy) | 0.9906 | 0.6857 |

**핵심 결과 — corruption 유형별 recall:**

| 모델 | Entity corruption recall | Relation corruption recall | 격차 |
|---|---|---|---|
| BERT (fine-tuned) | 0.9965 | 0.9808 | +0.0157 |
| TF-IDF baseline | 0.7694 | 0.5817 | +0.1878 |

**가설은 두 모델 모두에서 성립한다.** BERT는 격차가 작다(더 어려운 상황에서도
relation corruption을 잘 탐지한다). 반면 TF-IDF는 격차가 크다:
`entity_corruption`의 평균 코사인 유사도(0.104)는 정상 쌍(0.178)보다 뚜렷이
낮지만, `relation_corruption`(0.166)은 정상 쌍과 거의 구분되지 않는다 —
relation 이름(`cityServed`, `foundedBy` 등)은 맞든 틀리든 문장에 문자 그대로
등장하는 일이 드물어서, 어휘 중첩 기반 베이스라인은 활용할 신호가 거의 없기
때문이다. 전체 분해, 정성적 오류 사례, 보조 패턴(숫자 object의 entity
corruption이 명명된 entity corruption보다 어렵다는 점)에 대한 설명은
[`data/errors/error_analysis_report.md`](data/errors/error_analysis_report.md)에
있다.

## 스코프

의도적으로 제외했고, 향후 과제로 남겨둔 것들:
- 다른 노이즈 유형(트리플 순서 오류, 다중 트리플 혼합, 암묵적 relation 노이즈).
- 실제 distant-supervision 파이프라인 — 이 프로젝트는 깨끗한 WebNLG에 합성
  노이즈를 주입하는 방식이고, 실제로 노이즈가 섞인 데이터(예: NYT-FB)를
  다루지 않는다. 1~8단계는 실제 distant-supervision 데이터에 대한 accuracy를
  주장하지 않는다. 그 수치를 아예 낼 수 없는 이유는 아래 부록 참고.

## 부록 — NYT-FB 정성적 탐침 (일반화 벤치마크 아님)

[`scripts/09_nyt_case_study.py`](scripts/09_nyt_case_study.py)는 WebNLG로만
학습된 BERT 모델을 NYT-FB(`xiaobendanyn/nyt10`)에서 무작위로 뽑은 30개
문장에 대해 zero-shot으로 돌린다. 두 가지 트리플 표기 방식 — 원본 Freebase
relation 경로(`/people/person/nationality`)와 단순 자연어 변환(경로 마지막
토큰만, 예: `nationality`) — 로 각각 추론해서, relation *표기*만으로
예측이 달라지는지 본다.

**이건 accuracy 수치를 내지 않는다.** NYT-FB는 distant-supervision 데이터라
"이 문장이 실제로 이 트리플을 나타내는가"에 대한 사람 검증 라벨이 없다.
그래서 예측을 채점할 기준 자체가 없다. 출력물
(`data/errors/nyt_case_study_report.md`)은 사람이 직접 사례를 고르기 위한
예측 나열 표일 뿐, 지표가 아니다.

읽기 전에 알아둘 것 두 가지:
- **모델은 30건 중 29건에서 "noisy"로 예측이 쏠렸다**, 두 표기 방식 모두에서
  (표기 간 예측이 뒤집힌 건 0건). 유일한 예외는 relation 단어가 문장에 문자
  그대로 등장한 사례였다("... founders of Endemol ..." / relation
  `founders`) — 8단계에서 확인한 "모델이 relation 토큰과 문장의 어휘
  중첩에 의존한다"는 결과와 일치한다.
- **이게 단순히 truncation 때문 아닐까?** `max_length=48`은 훨씬 짧은
  WebNLG 문장에 맞춰 정해졌는데(4단계), NYT 문장 30건 중 25건이 이 길이를
  초과한다 — 일부는 길이의 절반 이상을 초과한다.
  [`scripts/10_nyt_notrunc_comparison.py`](scripts/10_nyt_notrunc_comparison.py)가
  같은 30건을 `max_length=128`(이 세트에서는 truncation이 거의 없음)로
  재추론해서 확인했다. **결과: 예측 라벨은 하나도 바뀌지 않았다**(두 표기
  모두 flip 0건, grounded 건수도 양쪽 다 1/30 그대로) — 즉 거의 전부
  "noisy"로 쏠리는 현상은 truncation 아티팩트가 아니다. 유일한 grounded
  사례의 *확신도*는 문장 전체가 보이자 꽤 떨어졌다(0.965 → 0.581) — 모델이
  주변 문맥이 늘어나면 희석되는 국소적 어휘 중첩에 의존한다는 것과
  일치한다. 2×2 전체 분해는
  [`data/errors/nyt_notrunc_comparison_report.md`](data/errors/nyt_notrunc_comparison_report.md)에
  있다. 주의: 모델은 고정된 `max_length=48` 배치로만 파인튜닝됐기 때문에,
  index 47 이후의 위치 임베딩은 파인튜닝 중 gradient를 한 번도 받은 적이
  없다 — 이 재추론은 진단용일 뿐, 모델이 length 128에서도 안정적으로
  동작한다는 근거는 아니다.

## 향후 과제

- corruption 유형을 확장해서(순서 오류, 다중 트리플 혼합, 암묵적 relation
  노이즈) 난이도 스펙트럼을 구축한다.
- 제대로 통제된 distant-supervision 일반화 검증에는 사람이 라벨링한
  소규모 테스트셋(NYT-FB 100~200건 수작업 주석)이 필요하다 — 위의 정성적
  탐침은 그것을 대신하는 게 아니라 임시 대체물이다.
- 검증된 탐지기를 KG 구축 파이프라인의 1차 노이즈 필터로 통합한다.
