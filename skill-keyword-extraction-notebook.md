# 2. 숙련 키워드 추출

이 노트북에서는 전처리된 구인공고 데이터에서 LLM을 활용하여 숙련 키워드를 추출하는 과정을 다룹니다.

## 2.1 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
from tqdm import tqdm
import openai
import time
import re

# OpenAI API 키 설정
openai.api_key = 'your-api-key-here'
```

## 2.2 전처리된 데이터 로드

```python
df = pd.read_csv('preprocessed_job_postings.csv')
print(f"Loaded {len(df)} preprocessed job postings")
```

## 2.3 LLM을 이용한 숙련 키워드 추출 함수 정의

```python
def extract_skills(text):
    prompt = f"""
    구인공고로부터 숙련을 추출하고 있습니다. 다음 구인공고에서 숙련 관련 키워드를 추출해주세요:

    {text}

    결과는 불필요한 응답 없이 단어만 제공하세요. 구분자 파이프 (|)로 구분하여 제공하세요.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 숙련을 추출하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return ""

# 샘플 데이터로 테스트
sample_text = df['preprocessed_text'].iloc[0]
print("Sample extracted skills:")
print(extract_skills(sample_text))
```

## 2.4 전체 데이터에 대한 숙련 키워드 추출

```python
# 주의: 이 과정은 시간이 많이 소요될 수 있습니다.
tqdm.pandas()
df['extracted_skills'] = df['preprocessed_text'].progress_apply(extract_skills)

# 결과 저장
df.to_csv('job_postings_with_skills.csv', index=False)
print("Saved job postings with extracted skills")
```

## 2.5 추출된 키워드 분석

```python
# 모든 추출된 스킬을 하나의 리스트로 모음
all_skills = [skill for skills in df['extracted_skills'].dropna() for skill in skills.split('|')]

# 가장 흔한 스킬 top 20
from collections import Counter

top_skills = Counter(all_skills).most_common(20)
print("Top 20 most common skills:")
for skill, count in top_skills:
    print(f"{skill}: {count}")

# 스킬 분포 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.bar([skill for skill, _ in top_skills], [count for _, count in top_skills])
plt.title('Top 20 Most Common Skills')
plt.xlabel('Skill')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

이 노트북에서는 LLM을 사용하여 구인공고에서 숙련 키워드를 추출하고, 추출된 키워드의 기본적인 분석을 수행했습니다. 다음 단계에서는 이 키워드들을 분류하고 정의를 생성하게 됩니다.
