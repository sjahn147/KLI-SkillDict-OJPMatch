# 6. 구인공고 매칭 시스템

이 노트북에서는 생성된 숙련 사전을 사용하여 구인공고와 매칭하는 시스템을 구현합니다.

# 구인공고(OJP) 숙련 매칭 프로세스

이 노트북은 구인공고(OJP) 데이터에 숙련 사전을 매칭하고, 결과를 처리하는 전체 프로세스를 설명합니다.

## 1. 프로젝트 개요

이 프로젝트는 다음과 같은 주요 단계로 구성됩니다:

1. 숙련 사전 로딩 및 전처리
2. 구인공고 데이터 처리 및 숙련 매칭 (p_file 생성)
3. 매칭 결과 확장 및 정리 (s_file 생성)

각 단계는 대용량 데이터를 효율적으로 처리하기 위해 멀티프로세싱을 활용합니다.

## 2. 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
import re
import os
import multiprocessing as mp
from tqdm import tqdm
from rapidfuzz import fuzz
import logging
import time
```

## 3. 숙련 사전 로딩 및 전처리

숙련 사전을 로드하고 정규표현식 패턴을 컴파일합니다.

```python
def load_skill_dict(file_path):
    df = pd.read_excel(file_path, usecols=['skill_code', 'skill_name', 'ext_key'])
    skill_dict = {}
    for _, row in df.iterrows():
        pattern = row['ext_key']
        if isinstance(pattern, str):
            if pattern.startswith('r\'') and pattern.endswith('\''):
                pattern = pattern[2:-1]
            if pattern.startswith('(?i)'):
                pattern = pattern[4:]
                flags = re.IGNORECASE
            else:
                flags = 0
            try:
                skill_dict[row['skill_code']] = {
                    'pattern': re.compile(pattern, flags),
                    'skill_name': row['skill_name']
                }
            except re.error:
                logging.error(f"Invalid regex pattern for skill_code {row['skill_code']}: {pattern}")
    return skill_dict

skill_dict = load_skill_dict("./skilldict/korskilldict.xlsx")
print(f"Loaded {len(skill_dict)} skills")
```

## 4. 구인공고 데이터 처리 및 숙련 매칭 (p_file 생성)

구인공고 텍스트를 전처리하고 숙련을 매칭합니다.

```python
def preprocess_job_desc(text):
    # 전처리 로직 (이전에 정의한 함수와 동일)
    ...

def match_skills(text, skill_dict, top_n=20, min_similarity=70):
    if pd.isna(text) or not isinstance(text, str) or text == "":
        return {}
    
    matched_skills = {}
    for skill_code, skill_info in skill_dict.items():
        if skill_info['pattern'].search(text):
            similarity = fuzz.partial_ratio(skill_info['skill_name'].lower(), text.lower())
            if similarity >= min_similarity:
                matched_skills[skill_code] = similarity
    
    return dict(sorted(matched_skills.items(), key=lambda x: x[1], reverse=True)[:top_n])

def process_job_file(file_path, skill_dict):
    df = pd.read_excel(file_path)
    df['preprocessed_text'] = df['pst_detail'].apply(preprocess_job_desc)
    df['skill_code'] = df['preprocessed_text'].apply(lambda x: ','.join(match_skills(x, skill_dict).keys()))
    return df

# 실제 처리는 멀티프로세싱을 사용하여 수행됩니다.
```

## 5. 매칭 결과 확장 및 정리 (s_file 생성)

p_file의 결과를 확장하고 정리하여 s_file을 생성합니다.

```python
def create_s_file(p_file, skill_dict_df):
    s_file = p_file.assign(skill_code=p_file['skill_code'].str.split(',')).explode('skill_code')
    s_file['skill_code'] = s_file['skill_code'].str.strip()
    s_file = s_file[s_file['skill_code'] != '']
    
    s_file = pd.merge(s_file, skill_dict_df[['skill_code', 'skill_name', 'main', 'sub', 'detail', 'softskill']], 
                      on='skill_code', how='left')
    
    columns_to_keep = ['skill_code', 'skill_name', 'main', 'sub', 'detail', 'softskill', 'idx', 'pst_id', 'company', 'bzno', 'snr', 'rnk', 'edu', 'type', 'sal', 'occ_code', 'keco_code']
    s_file = s_file[[col for col in columns_to_keep if col in s_file.columns]]
    
    if 'softskill' in s_file.columns:
        s_file['softskill'] = s_file['softskill'].replace({1: 'Y', 2: 'N'})
    
    return s_file

# 실제 처리는 멀티프로세싱을 사용하여 수행됩니다.
```

## 6. 전체 프로세스 실행

실제 프로젝트에서는 이 과정이 여러 개의 Python 스크립트로 나뉘어 있고, 멀티프로세싱을 사용하여 대규모 데이터를 처리합니다. 여기서는 간단한 예시로 프로세스의 흐름을 보여줍니다.

```python
def main():
    # 1. 숙련 사전 로드
    skill_dict = load_skill_dict("./skilldict/korskilldict.xlsx")
    
    # 2. p_file 생성 (실제로는 여러 파일을 병렬 처리)
    p_file = process_job_file("sample_job_posting.xlsx", skill_dict)
    p_file.to_excel("p_file_sample.xlsx", index=False)
    
    # 3. s_file 생성
    skill_dict_df = pd.read_excel("./skilldict/korskilldict.xlsx")
    s_file = create_s_file(p_file, skill_dict_df)
    s_file.to_excel("s_file_sample.xlsx", index=False)

if __name__ == "__main__":
    main()
```

이 노트북은 전체 프로세스의 개요를 제공하며, 각 단계의 핵심 로직을 보여줍니다. 실제 프로젝트에서는 대규모 데이터를 처리하기 위해 이 로직이 여러 스크립트로 나뉘어 있고 멀티프로세싱을 활용합니다.



## 6.1 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from fuzzywuzzy import fuzz
import multiprocessing
```

## 6.2 데이터 로드

```python
skill_dict = pd.read_csv('final_skill_dictionary.csv')
job_postings = pd.read_csv('preprocessed_job_postings.csv')

print(f"Loaded {len(skill_dict)} skills and {len(job_postings)} job postings")
```

## 6.3 정규표현식 기반 키워드 매칭 함수

```python
def create_regex_pattern(skill):
    # 스킬 키워드를 정규표현식 패턴으로 변환
    pattern = r'\b' + re.escape(skill.lower()) + r'\b'
    return re.compile(pattern)

# 스킬 사전의 모든 키워드에 대해 정규표현식 패턴 생성
skill_dict['regex_pattern'] = skill_dict['representative_skill'].apply(create_regex_pattern)

def match_skills(text, skill_patterns):
    matched_skills = []
    for skill_code, pattern in skill_patterns:
        if pattern.search(text.lower()):
            matched_skills.append(skill_code)
    return matched_skills

# 스킬 패턴 리스트 생성
skill_patterns = list(zip(skill_dict['skill_code'], skill_dict['regex_pattern']))
```

## 6.4 구인공고 매칭 함수

```python
def match_job_posting(row):
    text = row['preprocessed_text']
    matched_skills = match_skills(text, skill_patterns)
    return ', '.join(matched_skills)

# 멀티프로세싱을 위한 함수
def process_chunk(chunk):
    return chunk.apply(match_job_posting, axis=1)

# 멀티프로세싱을 사용한 매칭
def parallel_match_job_postings(df, num_processes=4):
    chunks = np.array_split(df, num_processes)
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(process_chunk, chunks)
    pool.close()
    pool.join()
    return pd.concat(results)

# 구인공고 매칭 실행
tqdm.pandas()
job_postings['matched_skills'] = parallel_match_job_postings(job_postings)

print("Sample of matched job postings:")
print(job_postings[['preprocessed_text', 'matched_skills']].head())
```

## 6.5 매칭 결과 분석

```python
# 매칭된 스킬 수 분석
job_postings['skill_count'] = job_postings['matched_skills'].apply(lambda x: len(x.split(', ')) if x else 0)

print("\nMatching statistics:")
print(f"Average number of skills per job posting: {job_postings['skill_count'].mean():.2f}")
print(f"Median number of skills per job posting: {job_postings['skill_count'].median():.2f}")
print(f"Job postings with no matched skills: {sum(job_postings['skill_count'] == 0)}")

# 가장 흔한 스킬 Top 10
all_matched_skills = [skill for skills in job_postings['matched_skills'].dropna() for skill in skills.split(', ')]
top_skills = pd.Series(all_matched_skills).value_counts().head(10)

print("\nTop 10 most common skills in job postings:")
print(top_skills)

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
top_skills.plot(kind='bar')
plt.title('Top 10 Most Common Skills in Job Postings')
plt.xlabel('Skill')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## 6.6 결과 저장

```python
job_postings.to_csv('job_postings_with_matched_skills.csv', index=False)
print("Saved job postings with matched skills")
```

이 노트북에서는 생성된 숙련 사전을 사용하여 구인공고와 매칭하는 시스템을 구현했습니다. 정규표현식 기반의 키워드 매칭을 사용하고, 멀티프로세싱을 통해 처리 속도를 개선했습니다. 또한 매칭 결과에 대한 기본적인 분석을 수행했습니다.
