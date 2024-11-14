## 2.1. 단어 통합

### 2.1.1. 숙련 단어 정의(skill term definition) 생성
  - 군집 분석을 위해서는 단어를 벡터화해서 거리를 계산해야함.
  - 그러나 단일 단어만으로는 단어의 의미를 정확하게 확정할 수 없음. 단어의 의미는 앞뒤 문맥 속에서 생성.
  - GPT 4-o mini 모델을 이용해 최초 추출한 raw skill term 33,800건에 대해 단어의 정의 생성함.

![image](https://github.com/user-attachments/assets/8da5c9df-468f-4a0c-88c7-511c6227e702)

### 2.1.2. 기계적 전처리
  - **공백 제거 및 타입 변환 :** 문장 분석을 할 때 공백이나 특수 문자는 오류를 유발할 수 있음. 정규식으로 이러한 텍스트를 제거하고, float 타입으로 인식될 수 있는 공백은 문자 타입(str)로 변환하였음.
  - **불용어 제거 :** "가능자', '가능', '능숙자', '능숙' ,'능력자','능력' 등의 의미 없지만 결과에 영향 주는 단어 제거.
  - **Lemmatization :** 영어 문장에서 단어의 품사 변형 (be/am/is/are/was/were…)은 실제 의미가 같음에도 다른 단어로 인식되어 거리 계산 결과에 영향을 줄 수 있음. 자연어 처리 라이브러리인 spaCy는 이러한 품사 변형을 제거하고 원형으로 통일하는 Lemmatization 방법을 제공함. 전처리 단계에서 리머타이즈까지 함께 진행하였음.
### 2.1.3. Sentence-BERT Embedding
  - 임베딩을 위한 다양한 선택지가 있지만, word2vec 같은 단순한 방법은 문장의 context를 반영하는 데 한계. 실제로 사용해보면 낮은 분류 성능을 나타냄. SBERT 방식은 문장 임베딩에 특화되어 있고 다국어를 지원하므로 분석 목적에 적합한 옵션이라고 할 수 있음.
  - 임베딩에 term과 definition을 모두 활용함. definition에 2배 가중치를 주도록 하여 생성된 문맥을 더 활용함.

```python
# SBERT 모델 로드
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
# 전처리 함수
def preprocess(text):
 if isinstance(text, float): # float인 경우 빈 문자열로 변환
 return ''
 text = str(text).strip().lower()
 words = nltk.word_tokenize(text)
 lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
 return ' '.join(lemmatized_words)
# Preprocess the keyword and definition columns
df['preprocessed_keyword'] = df['keyword'].apply(preprocess)
df['preprocessed_definition'] = df['definition'].apply(preprocess)
# 중복된 keyword 제거
df['preprocessed_keyword'] = df['preprocessed_keyword'].apply(lambda x: 
', '.join(set(x.split(', '))))
# Create a new column that combines preprocessed keyword and 
definition for SBERT embedding
df['keyword_with_definition'] = df['preprocessed_keyword'] + ' ' + 
df['preprocessed_definition']
```
### 2.1.4. 클러스터링
- 이 단계에서 클러스터링의 목적은 **유사 단어의 통합**임. 예를 들어, '3톤 지게차 운전', '3톤지게차운전', '지게차 운전능력' 등의 단어를 하나의 단어로 인식해 같은 숙련 코드를 부여하기 위한 거리 계산을 수행.
- 따라서 군집 계산의 결과 동일한 군집으로 판단되면 본 분석에서는 이들을 '같은 단어'로 보고 통합함.
- 클러스터링 알고리즘 선택을 위해 다음 방식들을 검토함:
    1. Agglomerative 방식: 미리 군집의 개수를 정하고 위계적으로 군집화하는 장점이 있으나, 노이즈 처리가 어려움
    2. K-means: 구현이 단순하나 클러스터 수를 사전에 지정해야 하는 제약이 있음
    3. DBSCAN: 클러스터 수를 미리 지정할 필요가 없고 노이즈 데이터를 자동으로 구분할 수 있음
- 숙련 키워드의 특성상 하위 범주 안에 몇 개의 유사 단어 군집이 있을지 미리 알 수 없고, 불규칙한 형태의 클러스터도 포착해야 하며, 관련 없는 키워드를 노이즈로 처리할 필요가 있으므로 **DBSCAN 방식을 채택**함.
### 2.1.5. 숙련코드 자동 할당
  - 클러스터링으로 그룹화된 유사 단어들에 동일한 숙련 코드를 자동 할당함
- **숙련 코드 구조** (총 9자리)
    1. 대분류 코드 (영어 대문자 2자리): 대분류명의 첫 두 글자
        - 예) '건축및건설' → 'AC'
    2. 하위분류 순번 (숫자 3자리)
    3. 숙련코드 순번 (숫자 4자리)
```
AC0010001
↳ AC: 건축및건설 (대분류)
↳ 001: 하위분류 순번
↳ 0001: 숙련코드 순번
```

#### 군집분석 및 숙련코드 할당 코드

```python
# 키워드와 정의를 벡터화하고 가중치를 적용 (SBERT 임베딩)
 keyword_vectors = model.encode(subset[＇preprocessed_keyword
＇].tolist())
definition_vectors = model.encode(subset[＇
keyword_with_definition＇].tolist())
combined_vectors = [combine_vectors(kv, dv, weight_keyword, 
weight_definition) for kv, dv in zip(keyword_vectors, definition_vectors)]
# 벡터 결합 함수 (정의에 2배 가중치)
def combine_vectors(vector1, vector2, weight1=1.0, weight2=2.0):
return weight1 * vector1 + weight2 * vector2
# 하위분류 코드를 3자리로 맞추기
def format_code2(x):
 if pd.isna(x) or not x.isdigit():
 return x
 return f"{int(x):03d}"

```

```python
# 하위분류마다 클러스터링 수행 및 스킬 코드 부여 (순차 처리)
def assign_skill_codes_sequential(df, model, weight_keyword=1.0, weight_definition=2.0, eps=0.30, min_samples=2, checkpoint_path=None):
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        df = checkpoint['df']
        processed_categories = checkpoint['processed_categories']
        total_length = checkpoint['total_length']
        processed_length = checkpoint['processed_length']
    else:
        df.loc[:, 'skill_code'] = '' # 'skill_code' 열 초기화
        df.loc[:, 'code2'] = '' # 'code2' 열 초기화
        processed_categories = set()
        total_length = len(df)
        processed_length = 0
    
    for category in tqdm(df['category'].unique(), desc="Processing Categories"):
        if category in processed_categories:
            continue
        for subcategory in tqdm(df[df['category'] == category]['subcategory'].unique(), desc=f"Processing Subcategories in {category}", leave=False):
            subset = df[(df['category'] == category) & (df['subcategory'] == subcategory)].copy()
            subset_indices = subset.index # 부분 집합의 인덱스 저장
            subcategory_skill_codes = [] # 서브카테고리 내에서 인덱스를 초기화
            
            if len(subset) < 2:
                logging.warning(f"Subcategory {subcategory} in {category} has less than 2 entries. Skipping clustering.")
                if len(subset) > 0 and 'code1' in subset.columns:
                    subcategory_code = format_code2(subcategory_codes.get(subcategory, "000"))
                    skill_code = f"{subset['code1'].iloc[0]}{subcategory_code}0000"
                    df.loc[subset_indices, 'skill_code'] = skill_code # 해당 인덱스에 skill_code 할당
                    df.loc[subset_indices, 'code2'] = subcategory_code # 해당 인덱스에 code2 할당
                    subcategory_skill_codes.append(skill_code)
                processed_length += len(subset)
                continue
            
            # 키워드와 정의를 벡터화하고 가중치를 적용 (SBERT 임베딩)
            keyword_vectors = model.encode(subset['preprocessed_keyword'].tolist())
            definition_vectors = model.encode(subset['keyword_with_definition'].tolist())
            combined_vectors = [combine_vectors(kv, dv, weight_keyword, weight_definition) for kv, dv in zip(keyword_vectors, definition_vectors)]
            
            # DBSCAN 클러스터링
            clustering_model_dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine') # 파라미터 값 불러옴
            cluster_assignment_dbscan = clustering_model_dbscan.fit_predict(combined_vectors)
            

```

사전에 지정한 subcategory별 클러스터링과 숙련코드 할당을 함께 진행함.

```python
            # 클러스터링 결과에 따라 순차적인 숙련코드 부여
            for cluster_id in np.unique(cluster_assignment_dbscan):
                if cluster_id == -1:
                    # 노이즈 포인트는 개별 숙련코드 부여
                    noise_indices = subset_indices[cluster_assignment_dbscan == -1]
                    for idx in noise_indices:
                        subcategory_code = format_code2(subcategory_codes.get(subcategory, "000"))
                        skill_code = f"{subset['code1'].iloc[0]}{subcategory_code}{len(subcategory_skill_codes):04d}"
                        df.loc[idx, 'skill_code'] = skill_code # 해당 인덱스에 skill_code 할당
                        df.loc[idx, 'code2'] = subcategory_code # 해당 인덱스에 code2 할당
                        subcategory_skill_codes.append(skill_code)
                else:
                    cluster_indices = subset_indices[cluster_assignment_dbscan == cluster_id]
                    subcategory_code = format_code2(subcategory_codes.get(subcategory, "000"))
                    skill_code = f"{subset['code1'].iloc[0]}{subcategory_code}{len(subcategory_skill_codes):04d}"
                    df.loc[cluster_indices, 'skill_code'] = skill_code # 해당 인덱스에 skill_code 할당
                    df.loc[cluster_indices, 'code2'] = subcategory_code # 해당 인덱스에 code2 할당
                    subcategory_skill_codes.append(skill_code)
            
            processed_length += len(subset)
        
        processed_categories.add(category)
        
        # 카테고리 처리 완료 후 checkpoint 저장
        if checkpoint_path:
            checkpoint = {
                'df': df,
                'processed_categories': processed_categories,
                'total_length': total_length,
                'processed_length': processed_length
            }
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint, f)
        
        logging.info(f"Processed {processed_length}/{total_length} entries.")
    
    return df['skill_code'].tolist()

# Checkpoint 파일 경로 설정
checkpoint_path = './checkpoint.pkl'
logging.info("Assigning skill codes...")

# 스킬 코드 부여
assign_skill_codes_sequential(df_sample, model, eps=0.30, min_samples=2, checkpoint_path=checkpoint_path)
```

### 2.1.6. 대표 레이블 지정 (군집별 퍼지 매칭)

![Pasted image 20241029232250](https://github.com/user-attachments/assets/07776e18-aa95-4cc9-8f12-1a2b12bfc22f)

- 앞선 단계에서 **군집분석에 의해 같은 클러스터 점수가 지정되면 반드시 '같은 단어＇로 보고 동일한 숙련 코드를 부여**하기로 하였음.
  - 동일한 숙련 코드를 부여받은 skill term들을 대표하는 하나의 단어를 고르기 위해 퍼지 매칭을 수행함. 이들은 실제 동일한지 여부와 상관없이 같은 단어로 보기로 하였으므로, **무조건 하나의 단어로 통합되도록 threshold 값을 조정**함.
  - 가능한 대표성이 높은 단어가 선택되도록, 출현 빈도가 가장 많은 단어의 전체 문자열을 representative keyword 열로 추가하였음. (정보 손실을 최소화하기 위함)
### 2.1.7. 숙련코드 기준 행 합치기

skill code를 기준으로 묶인 skill term들을 하나의 행으로 합치고, 기존의 skill term들을 하나의 열에 쉼표로 append하여 담는 aggregation을 진행.

#### 대표 레이블 지정 및 행 합치기 코드

```python
# Fuzzmatching을 사용한 대표 키워드 및 정의 식별
def fuzz_matching_representative(df, code_column, threshold=90, min_length=3):
    unique_codes = df[code_column].unique()
    representative_keywords = {}
    representative_definitions = {}
    for code in tqdm(unique_codes, desc=f"Fuzzmatching for {code_column}"):
        subset = df[df[code_column] == code]
        keywords = subset['keyword'].tolist()
        definitions = subset['preprocessed_definition'].tolist()
        if len(keywords) > 1:
            # 단어의 출현 빈도를 기준으로 대표 단어 선택
            keyword_counts = {}
            for keyword in keywords:
                if keyword not in keyword_counts:
                    keyword_counts[keyword] = 1
                else:
                    keyword_counts[keyword] += 1
            representative = max(keyword_counts, key=keyword_counts.get)
            representative_index = keywords.index(representative)
            representative_def = definitions[representative_index]
        elif len(keywords) == 1:
            representative = keywords[0]
            representative_def = definitions[0] if definitions else ""
        else:
            representative = ""
            representative_def = ""
        representative_keywords[code] = representative
        representative_definitions[code] = str(representative_def) if not pd.isna(representative_def) else ""
    return representative_keywords, representative_definitions

logging.info("Applying fuzz matching for DBSCAN results...")
# DBSCAN 결과에 대해 대표 키워드 및 정의 식별
rep_keywords_dbscan, rep_definitions_dbscan = fuzz_matching_representative(df_sample, 'skill_code', threshold=90, min_length=3)

# 대표 키워드와 정의를 새로운 열로 추가
df_sample['representative_keyword'] = df_sample['skill_code'].map(rep_keywords_dbscan)
df_sample['representative_definition'] = df_sample['skill_code'].map(rep_definitions_dbscan)

# 스킬코드에서 code3 추출하기
df_sample['code3'] = df_sample['skill_code'].str[-4:]

# 숙련코드 기준으로 keyword 통합
aggregated_df = df_sample.groupby('skill_code').agg({
    'category': 'first',
    'subcategory': 'first',
    'code1': 'first',
    'code2': 'first',
    'code3': 'first',
    'keyword': lambda x: ', '.join(str(keyword) for keyword in x if not pd.isna(keyword)),
    'representative_keyword': 'first',
    'representative_definition': 'first',
    'skill_yn': 'first',
    'softskill': 'first'
}).reset_index()
```

![Pasted image 20240907212315](https://github.com/user-attachments/assets/2046ecd0-1048-4983-bc43-60fcc99452b1)
