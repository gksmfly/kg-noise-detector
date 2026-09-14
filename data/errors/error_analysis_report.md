# Step 8: KG-to-Text 라벨 노이즈 탐지 - entity vs relation corruption 난이도 비교

## 핵심 결과: corruption_type별 recall

| Model | Entity corruption recall | Relation corruption recall | Gap |
|---|---|---|---|
| BERT (fine-tuned) | 0.9965 (1141/1145) | 0.9808 (1123/1145) | +0.0157 |
| TF-IDF baseline | 0.7694 (881/1145) | 0.5817 (666/1145) | +0.1878 |

## TF-IDF cosine similarity 평균 (corruption_type별)

이 값이 낮을수록 negative를 쉽게 걸러낼 수 있다는 뜻이다 (threshold보다 낮게 떨어지므로). positive와 비슷하게 높다면 표면 어휘로는 구분이 안 된다는 뜻이다.

| corruption_type | mean cosine similarity | n |
|---|---|---|
| none | 0.1781 | 1145 |
| entity | 0.1035 | 1145 |
| relation | 0.1663 | 1145 |

## 가설 검증 결과

가설("entity corruption이 relation corruption보다 탐지하기 쉽다")은 BERT와 TF-IDF 모두에서 지지됨.

- **TF-IDF에서 격차가 훨씬 크다** (entity 76.9% vs relation 58.2%, gap 18.8%p)는 예상한
  메커니즘을 그대로 보여준다. cosine similarity 평균을 보면 entity corruption은 평균
  0.1035로 positive(0.1781)보다 뚜렷하게 낮아 threshold로 쉽게 걸러지지만, relation
  corruption은 0.1663으로 positive와 거의 차이가 없다. 트리플 직렬화의 relation
  슬롯(예: `cityServed`, `foundedBy` 같은 카멜케이스 식별자)은 애초에 문장 표면에
  거의 등장하지 않으므로, 정답이든 오답이든 relation 토큰이 sentence와 겹치는 정도가
  비슷해 TF-IDF는 사실상 relation corruption 여부를 볼 수 있는 신호 자체가 부족하다.
- **BERT에서도 같은 방향의 격차가 있지만(1.6%p) 훨씬 작다** — 사전학습된 문맥 표현
  덕분에 relation corruption도 상당 부분 잡아내지만(recall 98.1%), 그래도 entity
  corruption(99.65%)보다는 체계적으로 더 어렵다는 것을 확인할 수 있다.
- **BERT의 entity_corruption 오분류(4건) 중 3건은 object가 숫자인 경우였다**
  (예: "mass 9.7" -> "7.5 kg", "playerNumber Tomato" -> "50"). 사람/지명 같은 엔티티를
  다른 엔티티로 바꾸면 표면적으로 명백히 이상하지만, 숫자를 다른 숫자로 바꾸는 건
  모델이 외부 사실 지식 없이는 "9.7이 맞는지 7.5가 맞는지" 판단할 근거가 텍스트
  안에 없다 — 이 자체가 entity corruption 안에서도 "엔티티 종류에 따라 난이도가
  갈린다"는 세부 패턴을 보여준다 (향후 과제로 확장할 만한 지점).
- **BERT의 relation_corruption 오분류(22건) 다수는 치환된 relation이 우연히 문장과
  의미적으로 그럴듯하게 들어맞은 경우였다** (예: 실제로는 다른 관계인데 `foundedBy`로
  치환됐고 문장이 "was founded"라고 서술하는 사례 - 관계명의 어근이 문장 표현과 겹침).
  즉 relation corruption의 어려움은 두 가지가 겹친 결과다: (1) 관계명 자체가 문장에
  잘 드러나지 않고, (2) 무작위 치환이 가끔 의미적으로 그럴듯한 관계를 뽑아 진짜
  hard negative를 만든다.

## BERT: relation_corruption을 grounded로 오판한 사례 (확신도 높은 순, 최대 15건)

### 사례 1 (pair_id=webnlg:5686, P(grounded)=0.999)
- triple_text: 14th_New_Jersey_Volunteer_Infantry_Monument | foundedBy | 1907-07-11
- sentence: 14th New Jersey Volunteer Infantry Monument was founded 1907-07-11.

### 사례 2 (pair_id=webnlg:1074, P(grounded)=0.999)
- triple_text: Allen_Forrest | activeYearsStartDate | 2005
- sentence: Allen Forrest became active in 2005.

### 사례 3 (pair_id=webnlg:7215, P(grounded)=0.999)
- triple_text: ACM_Transactions_on_Information_Systems | isPartOf | "1558-2868"
- sentence: The ISSN number of ACM Transactions on Information Systems is 1558-2868.

### 사례 4 (pair_id=webnlg:3466, P(grounded)=0.999)
- triple_text: Anderson_Township,_Madison_County,_Indiana | region | United_States
- sentence: The country of Anderson Township, Madison County, Indiana, is United States.

### 사례 5 (pair_id=webnlg:654, P(grounded)=0.998)
- triple_text: Saranac_Lake,_New_York | state | Essex_County,_New_York
- sentence: Saranac Lake, New York is part of Essex County, New York.

### 사례 6 (pair_id=webnlg:187, P(grounded)=0.998)
- triple_text: Alderney | leaderParty | Elizabeth_II
- sentence: The leader's name of Alderney is Elizabeth II.

### 사례 7 (pair_id=webnlg:2371, P(grounded)=0.998)
- triple_text: 200_Public_Square | cityServed | Cleveland
- sentence: 200 Public square is in Cleveland.

### 사례 8 (pair_id=webnlg:1641, P(grounded)=0.998)
- triple_text: William_Anders | position | 1963
- sentence: William Anders was chosen by NASA in 1963.

### 사례 9 (pair_id=webnlg:1610, P(grounded)=0.998)
- triple_text: William_Anders | origin | United_States
- sentence: The nationality of William Anders is United States.

### 사례 10 (pair_id=webnlg:66, P(grounded)=0.997)
- triple_text: Adirondack_Regional_Airport | regionServed | "SLK"
- sentence: Adirondack Regional Airport location identifier is SLK.

### 사례 11 (pair_id=webnlg:62, P(grounded)=0.997)
- triple_text: Adirondack_Regional_Airport | 1stRunwaySurfaceType | 6573
- sentence: The 1st runway length in feet of Adirondack Regional Airport is 6573.

### 사례 12 (pair_id=webnlg:3173, P(grounded)=0.995)
- triple_text: 1101_Clematis | bodyStyle | 5.7 (kilograms)
- sentence: 1101 Clematis has a mass of 5.7 kilograms.

### 사례 13 (pair_id=webnlg:1128, P(grounded)=0.995)
- triple_text: Anders_Osborne | musicFusionGenre | Rock_music
- sentence: Anders Osborne is an exponent of Rock music.

### 사례 14 (pair_id=webnlg:47, P(grounded)=0.992)
- triple_text: Abilene_Regional_Airport | regionServed | "ABI"
- sentence: The location Identifier of Abilene Regional Airport is ABI.

### 사례 15 (pair_id=webnlg:4138, P(grounded)=0.987)
- triple_text: Chinabank | isPartOf | Banking
- sentence: Chinabank's service is banking.


## BERT: entity_corruption을 grounded로 오판한 사례 (확신도 높은 순, 최대 15건)

### 사례 1 (pair_id=webnlg:5711, P(grounded)=0.998)
- triple_text: Adams_Township,_Madison_County,_Indiana | hasToItsWest | Franklin_County,_Pennsylvania
- sentence: Adams County Pennsylvania is East of Franklin County, Pennsylvania.

### 사례 2 (pair_id=webnlg:3131, P(grounded)=0.996)
- triple_text: 109_Felicitas | mass | 9.7
- sentence: 109 Felicitas has 7.5 kg of mass.

### 사례 3 (pair_id=webnlg:1854, P(grounded)=0.921)
- triple_text: Akeem_Dent | playerNumber | Tomato
- sentence: Akeem Dent is a player number 50.

### 사례 4 (pair_id=webnlg:3173, P(grounded)=0.663)
- triple_text: 1101_Clematis | mass | 1.2 (litres)
- sentence: 1101 Clematis has a mass of 5.7 kilograms.


## category별 오분류 분포 (참고)

- 전체 category 분포: {'Airport': 327, 'MeanOfTransportation': 309, 'Athlete': 309, 'Politician': 291, 'Artist': 285, 'Food': 273, 'City': 270, 'SportsTeam': 237, 'Building': 228, 'WrittenWork': 219}
- 오분류(noisy->grounded) category 분포: {'Airport': 5, 'CelestialBody': 3, 'MeanOfTransportation': 3, 'Monument': 2, 'Artist': 2, 'City': 2, 'Astronaut': 2, 'ComicsCharacter': 2, 'WrittenWork': 1, 'Building': 1}
- BERT entity corruption 오분류 4건 / relation corruption 오분류 22건