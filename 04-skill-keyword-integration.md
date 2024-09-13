

- 대표 레이블 지정 (각 군집 강제 퍼지 매칭)
  - 앞선 단계에서 군집분석에 의해 같은 클러스터 점수가 지정되면 '같은 단어＇로 보고 동일한 숙련 코드를 부여하기로 하였음.
  - 동일한 숙련 코드를 부여받은 skill term들을 대표하는 하나의 단어를 고르기 위해 강제 퍼지 매칭을 수행함. 이들은 실제 동일한지 여부와 상관없이 같은 단어로 보기로 하였으므로, 무조건 하나의 단어로 통합되도록 threshold 값을 조정함.
  - 가능한 대표성이 높은 단어가 선택되도록, 출현 빈도가 가장 많은 단어의 전체 문자열을 representative keyword 열로 추가하였음. (정보 손실을 최소화하기 위함)

- 숙련코드 기준 행 합치기
  - skill code를 기준으로 묶인 skill term들을 하나의 행으로 합치고, 기존의 skill term들을 하나의 열에 쉼표로 append하여 담는 aggregation을 진행.

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
