# 3. 숙련 분류 및 정의 생성

이 노트북에서는 추출된 숙련 키워드를 분류하고 각 숙련에 대한 정의를 생성하는 과정을 다룹니다.

## 3.1 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
from tqdm import tqdm
import openai

# OpenAI API 키 설정
openai.api_key = 'your-api-key-here'
```

## 3.2 추출된 숙련 키워드 데이터 로드

```python
df = pd.read_csv('job_postings_with_skills.csv')
print(f"Loaded {len(df)} job postings with extracted skills")

# 중복 제거된 유니크한 스킬 리스트 생성
unique_skills = set([skill.strip() for skills in df['extracted_skills'].dropna() for skill in skills.split('|')])
print(f"Total unique skills: {len(unique_skills)}")
```

## 3.3 LLM을 이용한 숙련 분류 및 정의 생성 함수

```python
def classify_and_define_skill(skill):
    prompt = f"""
    다음 숙련을 분류하고 정의를 생성해주세요:

    숙련: {skill}

    다음 형식으로 응답해주세요:
    category|subcategory|skill|keyword|skill_yn|softskill|definition

    category: 숙련의 대분류
    subcategory: 숙련의 하위분류
    skill: 숙련의 이름
    keyword: skill을 조금 더 일관성있고 정돈된 단어로 정리한 것
    skill_yn: 1-숙련, 2-숙련 아님, 3-자격증, 4-학력, 5-식별 불가
    softskill: 1-소프트 숙련, 2-하드 숙련
    definition: 숙련에 대한 간단한 정의
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "당신은 숙련을 분류하고 정의하는 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return ""

# 샘플 스킬로 테스트
sample_skill = list(unique_skills)[0]
print("Sample classification and definition:")
print(classify_and_define_skill(sample_skill))
```

## 3.4 모든 유니크 스킬에 대한 분류 및 정의 생성

```python
# 주의: 이 과정은 시간이 많이 소요될 수 있습니다.
results = []
for skill in tqdm(unique_skills):
    result = classify_and_define_skill(skill)
    if result:
        results.append(result.split('|'))

# 결과를 DataFrame으로 변환
classified_skills = pd.DataFrame(results, columns=['category', 'subcategory', 'skill', 'keyword', 'skill_yn', 'softskill', 'definition'])

# 결과 저장
classified_skills.to_csv('classified_skills.csv', index=False)
print("Saved classified skills")
```

## 3.5 분류 결과 분석

```python
print("Skills distribution:")
print(classified_skills['skill_yn'].value_counts())

print("\nSoft skills vs Hard skills:")
print(classified_skills['softskill'].value_counts())

print("\nTop 10 categories:")
print(classified_skills['category'].value_counts().head(10))

# 시각화
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
classified_skills['category'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Skill Categories')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

이 노트북에서는 LLM을 사용하여 추출된 숙련 키워드를 분류하고 정의를 생성했습니다. 또한 분류 결과에 대한 기본적인 분석을 수행했습니다. 다음 단계에서는 이 분류된 숙련들을 클러스터링하게 됩니다.
