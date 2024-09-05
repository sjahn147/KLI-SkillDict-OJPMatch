

### 단어 통합 절차
- 숙련 단어 정의(skill term definition) 생성
  - 군집 분석을 위해서는 단어를 숫자화(digitize)해서 거리를 계산해야함.
  - 그러나 단일 단어만으로는 단어의 의미를 정확하게 확정할 수 없음. 단어의 의미는 앞뒤 문맥 속에서 생성.
  - GPT 3.5 Instruct 모델을 이용해 최초 추출한 raw skill term 33,800건에 대해 단어의 정의 생성함.
![Pasted image 20240907213800](https://github.com/user-attachments/assets/5cad0b8d-a465-49e2-9c0f-e3a0f86e2fc5)

- 기계적 전처리 및 Lemmatization
  - 문장 분석을 할 때 공백이나 특수 문자는 오류를 유발할 수 있음. 정규식으로 이러한 텍스트를 제거하고, float 타입으로 인식될 수 있는 공백은 문자 타입(str)로 강제 변환하였음.
  - "가능자', '가능', '능숙자', '능숙' ,'능력자','능력' 등의 의미 없지만 결과에 영향 주는 단어 제거.
  - 영어 문장에서 단어의 품사 변형 (be/am/is/are/was/were…)은 실제 의미가 같음에도 다른 단어로 인식되어 거리 계산 결과에 영향을 줄 수 있음. 자연어 처리 라이브러리인 spaCy는 이러한 품사 변형을 제거하고 원형으로 통일하는 Lemmatization 방법을 제공함. 전처리 단계에서 리머타이즈까지 함께 진행하였음.

- Sentence BERT Embedding
  - 임베딩을 위한 다양한 선택지가 있지만, word2vec 같은 단순한 방법은 문장의 context를 반영하는 데 한계. 실제로 사용해보면 낮은 분류 성능을 나타냄. SBERT 방식은 문장 임베딩에 특화되어 있고 다국어를 지원하므로 분석 목적에 적합한 옵션이라고 할 수 있음.
  - 임베딩에 term과 definition을 모두 활용함. definition에 2배 가중치를 주도록 하여 생성된 문맥을 더 활용함.

- DBSCAN 클러스터링
  - 이 단계에서 클러스터링의 목적은 유사 단어의 통합임. 예를 들어, '3톤 지게차 운전', '3톤지게차운 전', '지게차 운전능력' 등의 단어를 하나의 단어로 인식해 같은 숙련 코드를 부여하기 위한 거리 계산을 수행.
  - 따라서 군집 계산의 결과 동일한 군집으로 판단되면 본 분석에서는 이들을 '같은 단어＇로 보고 통합함.
  - 클러스터링을 위해 미리 군집의 개수를 정하고 위계적으로 군집화하는 agglomerative 방식과 사전에 군집의 개수를 정하지 않고 탐색적으로 군집화하는 DBSCAN 방식을 모두 검토함.
  - 하위 범주 안에 몇 개의 유사 단어 군집이 있을지 미리 알 수 없으므로, DBSCAN 방식을 채택함.
- 숙련 코드 부여
  - 클러스터링에 의해 같은 점수가 부여된 단어들에 같은 숙련 코드를 자동적으로 할당함.
  - 숙련 코드 생성 규칙은 대분류 두 자리 문자 코드 @@ + 하위 분류 순번 코드 세 자리 ### + 숙련코드 순번 코드 네 자리 #### 로 총 아홉 자리의 일련 코드가 생성됨. 대분류 문자 코드는 대분류의 앞 글자 두개를 따서 직접 부여하였음. 예 ) '건축및건설' 은 'AC'


#### 군집분석 및 숙련코드 할당 코드

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
