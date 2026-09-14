# NYT-FB 정성적 케이스 스터디 (zero-shot, WebNLG로만 학습된 BERT)

**주의**: 이 표는 accuracy가 아니다. NYT-FB(distant supervision)에는 "이 문장이 이 트리플을 실제로 나타내는가"에 대한 사람 검증 정답이 없다. 아래는 원본 Freebase 표기(`/people/person/nationality`)와 자연어 변환 표기(`nationality`) 각각으로 예측했을 때 결과가 어떻게 달라지는지 보여주는 정성적 비교일 뿐이다. **표기 차이로 예측이 뒤집힌(flipped=True) 사례부터 3~5개를 직접 읽고 케이스 스터디로 고를 것.**

- 표본 30건, 표기 방식에 따라 예측이 뒤집힌 사례 0건
- grounded 예측 비율: 원본표기 1/30 vs 자연어표기 1/30 (참고용 집계, 정답 없음)

**⚠️ truncation 교란 요인 — Step 10에서 분리 완료, 결론: 주된 원인 아님**:
`max_length=48`은 Step 4에서 WebNLG(train 평균 32토큰)를 기준으로 정한 값이라, NYT
문장은 훨씬 길어서 이 30건 중 **25건(83%)이 48토큰을 넘겨 잘렸다** (일부는 원래
94~102토큰). 이 결과가 truncation의 인공물일 가능성이 있어 [`scripts/10_nyt_notrunc_comparison.py`](../../scripts/10_nyt_notrunc_comparison.py)
로 같은 30건을 max_length=128(전부 안 잘림)로 재추론해 확인했다: **예측 라벨이
단 한 건도 바뀌지 않았다** (Freebase/자연어 표기 모두 flip 0건, grounded 건수도
1/30으로 동일). 즉 "거의 전부 noisy로 예측됨"은 truncation 때문이 아니라 문체·표기
이동 자체가 근본 원인이라는 근거가 됐다. 자세한 2x2 비교와 해석은
[`nyt_notrunc_comparison_report.md`](nyt_notrunc_comparison_report.md) 참고.

| flipped | sentence | subject \| relation(raw) \| object | 원본 pred (P=grounded) | subject \| relation(natural) \| object | 자연어 pred (P=grounded) |
|---|---|---|---|---|---|
| no | but both senator john mccain and senator barack obama have now had to express regret for saying that | `john mccain | /people/person/ethnicity | american` | noisy (0.001) | `john mccain | ethnicity | american` | noisy (0.001) |
| no | but the attacks have slowed markedly since the seizure of a public school in beslan in 2004 ended wi | `russia | /location/location/contains | beslan` | noisy (0.001) | `russia | contains | beslan` | noisy (0.001) |
| no | it has been an unusual tournament from the start for sharapova , who was two points from defeat agai | `camille pin | /people/person/nationality | france` | noisy (0.000) | `camille pin | nationality | france` | noisy (0.000) |
| no | she called repeatedly for israeli withdrawal from the west bank and gaza , and in recent years suppo | `israel | /location/location/contains | west bank` | noisy (0.001) | `israel | contains | west bank` | noisy (0.001) |
| no | those that have fired rockets into israel say they do so to respond to israeli military activities i | `israel | /location/location/contains | west bank` | noisy (0.001) | `israel | contains | west bank` | noisy (0.001) |
| no | ocean view is near bethany beach on the delaware coast . | `delaware | /location/location/contains | ocean view` | noisy (0.029) | `delaware | contains | ocean view` | noisy (0.177) |
| no | iran said last year that it planned to install 3,000 centrifuges by this spring at natanz , as a fir | `iran | /location/location/contains | natanz` | noisy (0.001) | `iran | contains | natanz` | noisy (0.002) |
| no | under the leadership of his son tyrrhenus , the emigrating lydians built ships , loaded all the stor | `umbria | /location/administrative_division/country | italy` | noisy (0.001) | `umbria | country | italy` | noisy (0.000) |
| no | the senators have been joined in their effort by the republican leader , mitch mcconnell of kentucky | `mitch mcconnell | /people/person/place_lived | kentucky` | noisy (0.001) | `mitch mcconnell | place lived | kentucky` | noisy (0.000) |
| no | -lrb- ap -rrb- cavaliers 110 , hawks 76 -- lebron james scored 23 points , sasha pavlovic added 15 a | `lebron james | /people/person/place_lived | cleveland` | noisy (0.000) | `lebron james | place lived | cleveland` | noisy (0.001) |
| no | mr. sarkozy , who was campaigning on saturday with labor minister jean-louis borloo in northern fran | `jean-louis borloo | /people/person/nationality | france` | noisy (0.000) | `jean-louis borloo | nationality | france` | noisy (0.000) |
| no | others who have already indicated they will wear no. 42 include ken griffey jr. of cincinnati , flor | `gary sheffield | /people/person/place_lived | florida` | noisy (0.001) | `gary sheffield | place lived | florida` | noisy (0.000) |
| no | deal for ` no deal ' creator a group of investors led by john de mol , one of the founders of endemo | `endemol | /business/company/founders | john de mol` | grounded (0.965) | `endemol | founders | john de mol` | grounded (0.858) |
| no | the white house took the unusual step of having mr. bartlett conduct a hurried briefing with reporte | `mexico | /location/location/contains | mérida` | noisy (0.001) | `mexico | contains | mérida` | noisy (0.001) |
| no | stolzer parkhaus has built 28 automated garages in 11 countries since its first , in kronach , germa | `germany | /location/location/contains | kronach` | noisy (0.000) | `germany | contains | kronach` | noisy (0.000) |
| no | seth goldman , the president and chief executive of honest tea , which makes organic teas and juices | `honest tea | /business/company/founders | seth goldman` | noisy (0.000) | `honest tea | founders | seth goldman` | noisy (0.000) |
| no | israeli officials have rejected the idea of extending the cease-fire to the west bank unless it is p | `israel | /location/location/contains | west bank` | noisy (0.002) | `israel | contains | west bank` | noisy (0.001) |
| no | for the uninitiated , the annual orange battles of ivrea in northern italy are a lesson in both phys | `italy | /location/location/contains | ivrea` | noisy (0.001) | `italy | contains | ivrea` | noisy (0.003) |
| no | he saw service during world war ii in the middle east and italy and , after india became independent | `india | /location/location/contains | jammu` | noisy (0.001) | `india | contains | jammu` | noisy (0.001) |
| no | france , where ms. satrapi has lived for much of her adult life , does not figure much in the narrat | `paris | /location/administrative_division/country | france` | noisy (0.000) | `paris | country | france` | noisy (0.009) |
| no | dozens of moderate politicians and several conservatives gathered last week at the office of mehdi k | `mehdi karroubi | /people/person/nationality | iran` | noisy (0.000) | `mehdi karroubi | nationality | iran` | noisy (0.001) |
| no | she was hidden by a family in central france during the war , and returned after liberation to her s | `france | /location/location/contains | strasbourg` | noisy (0.001) | `france | contains | strasbourg` | noisy (0.001) |
| no | it remained the administrative boundary separating israel from the occupied territories , with one l | `israel | /location/location/contains | west bank` | noisy (0.001) | `israel | contains | west bank` | noisy (0.001) |
| no | devoted widow of the late james a. dobkin , she was the wise , loving and vibrant head of her own fa | `maryland | /location/location/contains | bethesda` | noisy (0.000) | `maryland | contains | bethesda` | noisy (0.001) |
| no | recently , one customs curmudgeon relieved me of some drop-dead delicious fried pork rinds from jaén | `spain | /location/location/contains | jaén` | noisy (0.001) | `spain | contains | jaén` | noisy (0.004) |
| no | jordan ruled the west bank from the armistice that concluded the 1948-49 war until 1967 , when israe | `israel | /location/location/contains | west bank` | noisy (0.006) | `israel | contains | west bank` | noisy (0.034) |
| no | in the oxycontin case , anderson kill arranged with james r. fogarty , a greenwich lawyer , to act a | `connecticut | /location/location/contains | greenwich` | noisy (0.001) | `connecticut | contains | greenwich` | noisy (0.001) |
| no | one of the biggest and best known north american winter festivals , the winterlude in ottawa and acr | `gatineau | /location/administrative_division/country | canada` | noisy (0.000) | `gatineau | country | canada` | noisy (0.004) |
| no | palestinian moderates in gaza have voiced concern recently over what they call the growth of '' al q | `gaza strip | /location/country/capital | gaza` | noisy (0.001) | `gaza strip | capital | gaza` | noisy (0.001) |
| no | senegal has created new universities in provincial capitals like saint louis and ziguinchor , but fe | `senegal | /location/location/contains | ziguinchor` | noisy (0.000) | `senegal | contains | ziguinchor` | noisy (0.000) |

## 케이스 스터디 (초안 — 직접 검토 후 슬라이드용으로 다듬을 것)

- [x] **사례 1 (relation 단어가 문장에 그대로 등장 → grounded로 확신, 30건 중 유일한 성공 사례)**
  트리플 `endemol | /business/company/founders | john de mol`, 문장은 "... john de mol,
  one of the **founders** of endemol, ..." — 원본표기 P(grounded)=0.965, 자연어표기
  P=0.858로 둘 다 높다. relation 경로의 마지막 단어("founders")가 문장에 정확히
  등장하는 유일한 사례이기도 하다. WebNLG 학습 때부터 있었던 경향(관계명 어휘가
  문장에 겹치면 grounded로 강하게 판단)이 완전히 다른 도메인에서도 그대로 재현된다 —
  "정말 관계를 이해해서" 판단하는 게 아니라 "관계명 단어가 문장에 있는지"에 크게
  기대는 shortcut일 가능성을 시사한다.
- [x] **사례 2 (표기 차이로 확신도는 크게 움직였지만 예측 라벨 자체는 안 뒤집힌 경계 사례)**
  트리플 `delaware | contains | ocean view`, 문장 "ocean view is near bethany beach on
  the delaware coast." 원본표기 P=0.029 -> 자연어표기 P=0.177로 6배 가까이 뛰었지만
  둘 다 noisy 판정. "contains"라는 흔한 영어 단어가 Freebase 경로보다 모델에게
  더 "그럴듯한 relation처럼" 읽힌다는 뜻으로 보인다 — 완전히 뒤집히진 않았지만
  자연어 변환이 방향성은 있다는 근거로 쓸 수 있다.
- [x] **사례 3 (truncation이 실제로 정보를 지웠지만, 그래도 라벨은 안 뒤집힌 사례 —
  "잘림 자체가 원인은 아니지만 문맥 길이가 확신도엔 영향을 준다")**
  트리플 `gatineau | country | canada`, 문장은 "... the winterlude in ottawa and across
  the ottawa river in gatineau, quebec, draws 650,000 people, including many from
  **outside canada**." (70토큰). 48-length로 자르면 정확히 `"...gatineau, quebe"`에서
  끊겨서, 트리플의 object와 문자 그대로 일치하는 단어 **"canada"가 통째로 사라진다**
  (직접 디코딩해서 확인함). 그 결과 자연어표기 P(grounded)가 0.0035(48) -> 0.0273(128)로
  약 8배 뛰었다 — 하지만 둘 다 여전히 noisy 판정이라 **최종 라벨은 안 바뀐다**.
  Step 10의 결론("truncation이 30건 전체의 noisy 쏠림을 만든 주된 원인은 아니다")과
  이 사례는 모순되지 않는다 — 오히려 정확히 그 결론이 뜻하는 바를 보여준다: 개별
  사례에서 truncation이 진짜로 증거 단어를 지우는 일은 실제로 일어나지만("canada"가
  사라짐), 그 효과의 크기(confidence 변화 0.003 -> 0.027)가 noisy/grounded 판정을
  뒤집을 만큼 크지는 않았다. 즉 truncation은 "잡음"이었지 "주된 원인"은 아니었다.