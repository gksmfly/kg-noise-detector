# Step 19: WebNLG vs DART(WikiSQL/WikiTableText) 도메인 일반화 비교

## 가설과 반대로 나온 핵심 결과

relation 이름이 WebNLG의 카멜케이스 KB 식별자(`cityServed`)보다 자연어에 훨씬 가까운 DART 표 컬럼명(`COLLEGE`, `CITY`)을 썼을 때, entity/relation 탐지 격차가 좁혀질 것으로 예상했다. **실제로는 정반대로 훨씬 크게 벌어졌다.**

| 도메인 | 모델 | Entity corruption recall | Relation corruption recall | Gap |
|---|---|---|---|---|
| WebNLG | BERT (fine-tuned) | 0.9965 | 0.9808 | +0.0157 |
| WebNLG | TF-IDF baseline | 0.7694 | 0.5817 | +0.1877 |
| DART | BERT (fine-tuned) | 0.9803 | 0.6930 | +0.2873 |
| DART | TF-IDF baseline | 0.8338 | 0.5239 | +0.3099 |

## 데이터 규모 비교 — 교란 요인 후보

DART로 도메인만 바꾼 게 아니라, 표본 크기와 relation당 예시 수도 함께 크게 줄었다. 이 두 요인이 뒤섞여 있어 '자연어성' 하나만의 효과라고 단정할 수 없다.

| | WebNLG | DART (WikiSQL/WikiTableText) | 비율 |
|---|---|---|---|
| positive 샘플 수 | 7,630 | 2,366 | 31% |
| train 행 수 | 16,023 | 4,968 | 31% |
| 고유 relation 수 | 346 | 1,066 | 308% |
| relation당 평균 예시 수 | 22.1 | 2.2 | 10% |

## 해석

가능한 설명 두 가지가 뒤섞여 있다:

1. **자연어성 가설이 틀렸다**: relation 이름이 자연어에 가까워도 recall 격차를 좁히지 못한다 — 표면 어휘 중첩과 무관하게 relation 검증 자체가 entity 검증보다 근본적으로 더 어려운 과제일 수 있다.
2. **데이터 희소성이 주된 원인**: DART는 relation당 평균 예시 수가 WebNLG의 약 10분의 1이다. Entity 판정(고유 문자열이 다른지 보는 국소적 판단)은 적은 데이터로도 배우기 쉽지만, relation 판정(트리플과 문장의 의미적 정합성 판단)은 더 많은 반복 노출이 필요한 과제라서, 데이터가 줄면 relation recall이 불균형하게 더 크게 떨어질 수 있다.

이 둘을 가르는 ablation: WebNLG train을 DART와 같은 표본 크기(4,968행)로 무작위 서브샘플링해서 Step 5와 동일하게 재학습하면 된다. relation 어휘의 자연어성은 WebNLG 그대로 유지한 채 표본 크기만 맞추는 대조군이므로, 거기서도 relation recall이 크게 떨어지면 (2)가 주된 원인이고, 여전히 격차가 작게 유지되면 (1)에 더 무게가 실린다.