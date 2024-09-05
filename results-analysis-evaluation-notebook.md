# 7. 결과 분석 및 평가

이 노트북에서는 구인공고 매칭 결과를 분석하고 시스템의 성능을 평가합니다.

## 7.1 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
```

## 7.2 데이터 로드

```python
job_postings = pd.read_csv('job_postings_with_matched_skills.csv')
skill_dict = pd.read_csv('final_skill_dictionary.csv')

print(f"Loaded {len(job_postings)} job postings and {len(skill_dict)} skills")
```

## 7.3 매칭 결과 분석

```python
# 스킬 매칭 분포
plt.figure(figsize=(12, 6))
job_postings['skill_count'].hist(bins=50)
plt.title('Distribution of Matched Skills per Job Posting')
plt.xlabel('Number of Matched Skills')
plt.ylabel('Count of Job Postings')
plt.show()

# 카테고리별 매칭 빈도
def get_skill_category(skill_code):
    return skill_dict[skill_dict['skill_code'] == skill_code]['category'].values[0] if skill_code in skill_dict['skill_code'].values else 'Unknown'

all_matched_skills = [skill for skills in job_postings['matched_skills'].dropna() for skill in skills.split(', ')]
skill_categories = [get_skill_category(skill) for skill in all_matched_skills]

category_counts = pd.Series(skill_categories).value_counts()

plt.figure(figsize=(12, 6))
category_counts.plot(kind='bar')
plt.title('Frequency of Matched Skill Categories')
plt.xlabel('Skill Category')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
```

## 7.4 매칭 정확도 평가 (수동 평가 샘플)

```python
# 주의: 이 부분은 실제로 수동 평가를 진행한 결과를 가정합니다.
# 실제 프로젝트에서는 전문가가 샘플 데이터에 대해 수동으로 평가한 결과를 사용해야 합니다.

def manual_evaluation(row):
    # 이 함수는 실제 수동 평가 결과를 시뮬레이션합니다.
    # 실제 프로젝트에서는 전문가의 평가 결과를 반영해야 합니다.
    matched_skills = set(row['matched_skills'].split(', ') if row['matched_skills'] else [])
    true_skills = set(row['true_skills'].split(', ') if row['true_skills'] else [])
    
    true_positives = len(matched_skills.intersection(true_skills))
    false_positives = len(matched_skills - true_skills)
    false_negatives = len(true_skills - matched_skills)
    
    return pd.Series({
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    })

# 샘플 데이터에 대한 수동 평가 시뮬레이션
sample_size = 100
sample_data = job_postings.sample(sample_size, random_state=42)
sample_data['true_skills'] = sample_data['matched_skills']  # 이 부분은 실제 수동 평가 결과로 대체되어야 합니다.

evaluation_results = sample_data.apply(manual_evaluation, axis=1)

# 정밀도, 재현율, F1 점수 계산
precision = evaluation_results['true_positives'].sum() / (evaluation_results['true_positives'].sum() + evaluation_results['false_positives'].sum())
recall = evaluation_results['true_positives'].sum() / (evaluation_results['true_positives'].sum() + evaluation_results['false_negatives'].sum())
f1_score = 2 * (precision * recall) / (precision + recall)

print("Matching Performance Metrics:")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
```

## 7.5 오차 분석

```python
# 가장 자주 발생하는 오류 분석 (실제 프로젝트에서는 수동 평가 결과를 기반으로 해야 합니다)
error_analysis = sample_data[sample_data.apply(lambda row: set(row['matched_skills'].split(', ')) != set(row['true_skills'].split(', ')), axis=1)]

print("\nSample of Error Cases:")
print(error_analysis[['preprocessed_text', 'matched_skills', 'true_skills']].head())
```

## 7.6 개선 방향 논의

```python
print("\nPotential Improvements:")
print("1. 정규표현식 패턴 개선: 더 정교한 패턴을 사용하여 false positive를 줄입니다.")
print("2. 컨텍스트 고려: 단순 키워드 매칭이 아닌 주변 문맥을 고려한 매칭 방식을 도입합니다.")
print("3. 동의어 처리: 같은 의미를 가진 다양한 표현을 처리