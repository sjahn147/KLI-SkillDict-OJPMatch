# 6. 구인공고 매칭 시스템

이 노트북에서는 생성된 숙련 사전을 사용하여 구인공고와 매칭하는 시스템을 구현합니다.

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
