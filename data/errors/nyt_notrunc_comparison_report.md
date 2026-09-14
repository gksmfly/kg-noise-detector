# Step 10: truncation 교란 요인 분리 (Step 9 후속)

**여전히 accuracy가 아니다.** NYT-FB에는 정답 라벨이 없다. 아래 비율은 "grounded로 예측된 건수"일 뿐 "맞게 예측한 건수"가 아니다.

**모델은 max_length=48로만 fine-tuning됐다.** 학습 중 모든 배치가 정확히 48 토큰으로 padding됐기 때문에, position 48~127에 대응하는 위치 임베딩은 fine-tuning 동안 한 번도 gradient를 받지 않았고 사전학습 BERT의 값 그대로다. 따라서 max_length=128 추론이 "더 정확해진다"는 보장은 전혀 없다 — 이 실험은 성능 개선을 확인하는 게 아니라, truncation 여부가 "거의 전부 noisy"라는 관찰에 얼마나 기여했는지를 분리해서 보려는 진단용 재추론이다.

- 30건 중 max_length=48에서 실제로 잘렸던 샘플: 25건
- max_length을 48->128로 바꿨을 때 예측이 바뀐 건수: Freebase표기 0건 / 자연어표기 0건

## 결론 (예상 질문 "그거 그냥 문장 잘려서 그런 거 아니에요?"에 대한 답)

**아니다.** 25/30건이 48-length에서 잘렸음에도, truncation을 풀어(128) 재추론했을 때
**예측 라벨이 단 한 건도 바뀌지 않았다** (Freebase 표기 0건, 자연어 표기 0건 flip).
grounded로 예측된 건수도 두 길이 모두 동일하게 1/30이다. 즉 Step 9에서 본
"거의 전부 noisy로 예측됨"은 truncation의 인공물이 아니라, 모델이 실제로 NYT-FB의
문체·표기 체계를 grounded로 인식하지 못한다는 더 근본적인 신호로 봐도 된다 —
truncation은 교란 요인이 아니었다.

다만 확률(확신도) 자체는 흥미롭게 움직인다. 유일한 grounded 사례("founders")는
128에서도 여전히 grounded로 남지만 확신도가 0.965 -> 0.581로 크게 떨어졌다 — 잘렸던
48-length 버전에서 오히려 "founders"라는 relation 단어가 상대적으로 더 두드러지게
보였는데, 문장 전체(84토큰)를 다 보여주자 그 주변의 무관한 문맥("a group of investors
led by...")이 신호를 희석시킨 것으로 보인다. 즉 "더 많은 문맥 = 더 확신"이 아니라
오히려 반대 방향으로 움직인 사례라, "관계명-문장 어휘 중첩에 의존한다"는 가설과도
결이 맞는다 (관련 없는 문맥이 늘어날수록 국소적인 어휘 신호가 상대적으로 약해짐).

## 2x2 비교표: grounded로 예측된 건수 (30건 중)

| | max_length=48 (학습 시, 원래 Step 9) | max_length=128 (재추론, truncation 거의 없음) |
|---|---|---|
| Freebase 표기 (`/people/person/nationality`) | 1/30 | 1/30 |
| 자연어 변환 표기 (`nationality`) | 1/30 | 1/30 |

## 해석 가이드

- **48->128에서 grounded 건수가 크게 늘었다면**: truncation이 "거의 전부 noisy" 관찰에 상당 부분 기여했다는 뜻 — Step 9의 결과를 "모델이 도메인 이동에 완전히 무너졌다"고 해석하면 과장이 된다.
- **128에서도 여전히 대부분 noisy라면**: truncation과 무관하게 표기/문체 이동 자체가 근본 원인이라는 더 강한 근거가 된다 — 이 경우 Step 9의 프레이밍이 정당화된다.
- 어느 쪽이든, 위에서 계산한 "48->128 예측 변화 건수"가 실질적 답이다. 이 숫자가 0에 가까우면 truncation은 주된 원인이 아니고, 크면 truncation이 상당 부분을 설명한다.

## founders 사례 재확인 (Step 9의 유일한 grounded 사례)

- `endemol | /business/company/founders | john de mol` (원문 길이 84토큰, 48에서 잘림)
  - max_length=48: Freebase P(grounded)=0.965, 자연어 P=0.858
  - max_length=128: Freebase P(grounded)=0.581, 자연어 P=0.737
  - relation 단어("founders")가 문장에 그대로 등장하는 이 사례가 128에서도 여전히 grounded로 남는지가, "관계명-문장 어휘 중첩" 패턴이 truncation과 무관하게 진짜인지 보여주는 핵심 체크포인트다.

## 30건 상세 (원문 토큰 길이·잘림 여부 포함)

| nyt_index | raw_len | 48서 잘림? | sentence | Freebase 48->128 | 자연어 48->128 |
|---|---|---|---|---|---|
| 3648 | 102 | YES | in the oxycontin case , anderson kill arranged with james r. fogarty , a greenwi | 0.001->0.000 | 0.001->0.001 |
| 3533 | 95 | YES | devoted widow of the late james a. dobkin , she was the wise , loving and vibran | 0.000->0.000 | 0.001->0.000 |
| 2006 | 94 | YES | -lrb- ap -rrb- cavaliers 110 , hawks 76 -- lebron james scored 23 points , sasha | 0.000->0.000 | 0.001->0.001 |
| 2144 | 88 | YES | others who have already indicated they will wear no. 42 include ken griffey jr.  | 0.001->0.001 | 0.000->0.000 |
| 2672 | 85 | YES | for the uninitiated , the annual orange battles of ivrea in northern italy are a | 0.001->0.001 | 0.003->0.004 |
| 2301 | 84 | YES | deal for ` no deal ' creator a group of investors led by john de mol , one of th | 0.965->0.581 | 0.858->0.737 |
| 1598 | 79 | YES | under the leadership of his son tyrrhenus , the emigrating lydians built ships , | 0.001->0.001 | 0.000->0.005 |
| 371 | 79 | YES | it has been an unusual tournament from the start for sharapova , who was two poi | 0.000->0.001 | 0.000->0.001 |
| 259 | 79 | YES | but the attacks have slowed markedly since the seizure of a public school in bes | 0.001->0.001 | 0.001->0.001 |
| 2537 | 75 | YES | seth goldman , the president and chief executive of honest tea , which makes org | 0.000->0.000 | 0.000->0.001 |
| 1825 | 71 | YES | the senators have been joined in their effort by the republican leader , mitch m | 0.001->0.001 | 0.000->0.000 |
| 3665 | 70 | YES | one of the biggest and best known north american winter festivals , the winterlu | 0.000->0.001 | 0.004->0.027 |
| 2074 | 66 | YES | mr. sarkozy , who was campaigning on saturday with labor minister jean-louis bor | 0.000->0.000 | 0.000->0.000 |
| 3544 | 66 | YES | recently , one customs curmudgeon relieved me of some drop-dead delicious fried  | 0.001->0.001 | 0.004->0.004 |
| 3963 | 66 | YES | senegal has created new universities in provincial capitals like saint louis and | 0.000->0.000 | 0.000->0.000 |
| 3258 | 64 | YES | it remained the administrative boundary separating israel from the occupied terr | 0.001->0.003 | 0.001->0.007 |
| 1199 | 62 | YES | iran said last year that it planned to install 3,000 centrifuges by this spring  | 0.001->0.001 | 0.002->0.001 |
| 2590 | 61 | YES | israeli officials have rejected the idea of extending the cease-fire to the west | 0.002->0.001 | 0.001->0.001 |
| 228 | 59 | YES | but both senator john mccain and senator barack obama have now had to express re | 0.001->0.001 | 0.001->0.001 |
| 2733 | 58 | YES | he saw service during world war ii in the middle east and italy and , after indi | 0.001->0.001 | 0.001->0.001 |
| 2791 | 58 | YES | dozens of moderate politicians and several conservatives gathered last week at t | 0.000->0.001 | 0.001->0.001 |
| 3666 | 56 | YES | palestinian moderates in gaza have voiced concern recently over what they call t | 0.001->0.001 | 0.001->0.001 |
| 2757 | 55 | YES | france , where ms. satrapi has lived for much of her adult life , does not figur | 0.000->0.001 | 0.009->0.009 |
| 668 | 53 | YES | she called repeatedly for israeli withdrawal from the west bank and gaza , and i | 0.001->0.001 | 0.001->0.001 |
| 3567 | 50 | YES | jordan ruled the west bank from the armistice that concluded the 1948-49 war unt | 0.006->0.033 | 0.034->0.034 |
| 2439 | 48 | no | the white house took the unusual step of having mr. bartlett conduct a hurried b | 0.001->0.001 | 0.001->0.001 |
| 2532 | 48 | no | stolzer parkhaus has built 28 automated garages in 11 countries since its first  | 0.000->0.000 | 0.000->0.000 |
| 2955 | 46 | no | she was hidden by a family in central france during the war , and returned after | 0.001->0.001 | 0.001->0.001 |
| 747 | 42 | no | those that have fired rockets into israel say they do so to respond to israeli m | 0.001->0.001 | 0.001->0.001 |
| 1137 | 31 | no | ocean view is near bethany beach on the delaware coast . | 0.029->0.029 | 0.177->0.177 |