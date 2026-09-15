# Step 18: DART(WikiSQL/WikiTableText) 도메인 일반화 탐침 - entity vs relation corruption 난이도

**배경**: WebNLG의 relation은 카멜케이스 KB 식별자(`cityServed`)라 문장 표면에 거의 등장하지 않는다. DART의 WikiSQL/WikiTableText 서브셋은 relation이 표 컬럼명(`COLLEGE`, `CITY`, `FEET`)이라 훨씬 자연어에 가깝다. 같은 corruption 방법론을 그대로 적용해, relation 이름의 자연어성이 entity/relation 탐지 격차를 좁히는지 검증한다.

## 핵심 결과: corruption_type별 recall

| Model | Entity corruption recall | Relation corruption recall | Gap |
|---|---|---|---|
| BERT (fine-tuned, DART) | 0.9803 (348/355) | 0.6930 (246/355) | +0.2873 |
| TF-IDF baseline (DART) | 0.8338 (296/355) | 0.5239 (186/355) | +0.3099 |

## TF-IDF cosine similarity 평균 (corruption_type별)

| corruption_type | mean cosine similarity | n |
|---|---|---|
| none | 0.4949 | 355 |
| entity | 0.2459 | 355 |
| relation | 0.4251 | 355 |

## WebNLG(Step 8) 대비 비교

| 도메인 | 모델 | Entity recall | Relation recall | Gap |
|---|---|---|---|---|
| WebNLG | BERT | 0.9965 | 0.9808 | +0.0157 |
| WebNLG | TF-IDF | 0.7694 | 0.5817 | +0.1878 |
| DART | BERT | 0.9803 | 0.6930 | +0.2873 |
| DART | TF-IDF | 0.8338 | 0.5239 | +0.3099 |

**예상과 반대 방향.** relation 이름이 더 자연어에 가까운 DART에서 격차가 좁혀질 것으로 예상했지만, 실제로는 BERT·TF-IDF 모두 DART에서 격차가 훨씬 크게 벌어졌다(BERT: 1.6%p -> 28.7%p, TF-IDF: 18.8%p -> 31.4%p 안팎). **중요한 교란 요인**: DART WikiSQL/WikiTableText 서브셋은 WebNLG보다 표본이 훨씬 작고(train 4,968행 vs 16,023행, 약 31%) relation 어휘는 훨씬 길게 꼬리를 문다(고유 relation 1,066개 vs 346개, 즉 relation당 평균 예시 수가 WebNLG는 약 22개인데 DART는 약 2.2개뿈이다). relation 판정은 entity 판정보다 더 많은 문맥적 학습이 필요한 과제일 수 있는데, 학습 데이터와 relation당 반복 노출이 둘 다 크게 줄어든 상태라 '자연어성 효과'와 '데이터 희소성 효과'가 뒤섞여 있다. 이 둘을 분리하려면 WebNLG train을 DART와 같은 규모로 서브샘플링해서 같은 실험을 반복하는 ablation이 필요하다(자연어성은 그대로 둔 채 표본 크기만 맞추는 대조군).

## BERT: relation_corruption을 grounded로 오판한 사례 (확신도 높은 순, 최대 15건)

### 사례 1 (pair_id=dart:2249, P(grounded)=0.885)
- triple_text: Nigel Benn | CIRCUIT | Champion boxer
- sentence: Nigel Benn was the champion boxer.

### 사례 2 (pair_id=dart:1969, P(grounded)=0.874)
- triple_text: Nathan Hoffart | DURATION | Saskatchewan
- sentence: Nathan Hoffart went to Saskatchewan College.

### 사례 3 (pair_id=dart:147, P(grounded)=0.855)
- triple_text: Sandy Kennon | TILL | Norwich City
- sentence: Sandy Kennon moved from Norwich City to Colchester United.

### 사례 4 (pair_id=dart:854, P(grounded)=0.855)
- triple_text: Team Lotus | GROSS | Trevor Taylor
- sentence: Trevor Taylor represented Team Lotus.

### 사례 5 (pair_id=dart:428, P(grounded)=0.850)
- triple_text: Maxime Dillies | WRITER(S) | P
- sentence: Maxime Dillies plays as a setter, or palleggiatore (P).

### 사례 6 (pair_id=dart:2191, P(grounded)=0.844)
- triple_text: Dominic Picard | FUNCTION | Laval
- sentence: Dominic Picard went to college at Laval.

### 사례 7 (pair_id=dart:452, P(grounded)=0.840)
- triple_text: Suntree Seniors Classic | PUBLISHER(S) | Miller Barber (2)
- sentence: Miller Barber won the Suntree Seniors Classic.

### 사례 8 (pair_id=dart:1147, P(grounded)=0.839)
- triple_text: Andrea Schöpp | COMMENT | 14
- sentence: Andrea schöpp was the skip with 14 stolen ends.

### 사례 9 (pair_id=dart:2177, P(grounded)=0.833)
- triple_text: Commerce Court West | SCORE | Toronto
- sentence: The Commerce Court West building was built in the city of Toronto.

### 사례 10 (pair_id=dart:1288, P(grounded)=0.831)
- triple_text: Silver Shadow Stakes | ORIGIN | swp
- sentence: With swp weight class the Silver Shadow Stakes was held.

### 사례 11 (pair_id=dart:492, P(grounded)=0.826)
- triple_text: Blinders | PERCENT | 2006
- sentence: Blinders was a 2006 program.

### 사례 12 (pair_id=dart:233, P(grounded)=0.819)
- triple_text: Eastern Glow | ELIMINATED_BY | 2.01
- sentence: The song was Eastern Glow and Epsiode for 2.01.

### 사례 13 (pair_id=dart:2288, P(grounded)=0.818)
- triple_text: Brazil | LANGUAGE | APC
- sentence: The Country is Brazil.The weapon description is APC.

### 사례 14 (pair_id=dart:1607, P(grounded)=0.808)
- triple_text: Saint Clare College of Caloocan | TECHNOLOGIES | Lady Saints
- sentence: Saint clare college of caloocan is represented by the lady saints.

### 사례 15 (pair_id=dart:1868, P(grounded)=0.805)
- triple_text: Indiana 4 | ORIGINAL_AIR_DATE | E. Ross Adair
- sentence: E. ross adair's the incumbent of Indiana 4 district.


## BERT: entity_corruption을 grounded로 오판한 사례 (확신도 높은 순, 최대 15건)

### 사례 1 (pair_id=dart:2230, P(grounded)=0.820)
- triple_text: July 12 | OPPONENT | Minnesota
- sentence: The game on July 12 played against was detroit.

### 사례 2 (pair_id=dart:322, P(grounded)=0.761)
- triple_text: PGE SA | RANK_IN_2011 | No
- sentence: PGE SA was ranked third in 2011.

### 사례 3 (pair_id=dart:2262, P(grounded)=0.726)
- triple_text: episode 4 | RUN_TIME | 24:01
- sentence: Episode Part Five had a run time of 24:01.

### 사례 4 (pair_id=dart:1537, P(grounded)=0.712)
- triple_text: 2003 | INFLATION_RATE_% | MA
- sentence: The inflation rate in 2003 was -2.5%.

### 사례 5 (pair_id=dart:1536, P(grounded)=0.692)
- triple_text: 1999 | INFLATION_RATE_% | 6,934,975
- sentence: Inflation decreased by 1.4 percent in 1999.

### 사례 6 (pair_id=dart:41, P(grounded)=0.614)
- triple_text: Anse la Raye | AREA_KM2 | texas
- sentence: Anse la Raye has 30.9 km2 land area

### 사례 7 (pair_id=dart:1675, P(grounded)=0.521)
- triple_text: Rwanda | TOTAL | Madagascar
- sentence: Rwanda won total 1 medals


## source(=category)별 오분류 분포 (참고)

- 전체 source 분포: {'WikiSQL_decl_sents': 582, 'WikiTableQuestions_lily': 273, 'WikiTableQuestions_mturk': 210}
- 오분류(noisy->grounded) source 분포: {'WikiSQL_decl_sents': 55, 'WikiTableQuestions_lily': 41, 'WikiTableQuestions_mturk': 20}
- BERT entity corruption 오분류 7건 / relation corruption 오분류 109건