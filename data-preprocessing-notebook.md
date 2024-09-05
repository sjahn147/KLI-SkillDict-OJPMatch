# 1. 데이터 전처리 및 탐색

이 노트북에서는 한국형 숙련사전 구축을 위한 구인공고 데이터의 전처리 및 탐색 과정을 다룹니다.

## 1.1 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from tqdm import tqdm

# 경고 메시지 무시
import warnings
warnings.filterwarnings('ignore')
```

## 1.2 데이터 로드

```python
def load_data(file_path):
    df = pd.read_excel(file_path)
    print(f"Loaded {len(df)} rows of data")
    return df

job_postings = load_data('구인공고_데이터.xlsx')
```

## 1.3 데이터 기본 정보 확인

```python
print(job_postings.info())
print("\nSample data:")
print(job_postings.head())
```

## 1.4 텍스트 전처리 함수 정의

```python
def preprocess_job_desc(text):
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        text = str(text)
    
    # 불필요한 문구 제거
    patterns_to_remove = [
        r"기업\s*소개\s*또는\s*채용\s*안내를\s*작성해\s*보세요\s*불필요시\s*'?소개글'?을\s*OFF하면\s*소개\s*영역이\s*숨겨집니다\.?",
        r"\*{3}\s*입사\s*지원",
        r"\*{3}\s*온라인\s*이력서",
        # ... (다른 패턴들)
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # 절사 기준 문구 인식 및 처리
    base_keywords = [
        r'혜택\s*및\s*복지',
        r'근무\s*조건',
        r'전형\s*절차',
        # ... (다른 키워드들)
    ]
    cut_keywords = '|'.join(base_keywords)
    pattern = fr'(^|\s|\S)({cut_keywords})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        text = text[:match.start() + 1]
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 전처리 적용
tqdm.pandas()
job_postings['preprocessed_text'] = job_postings['pst_detail'].progress_apply(preprocess_job_desc)
```

## 1.5 전처리 결과 확인

```python
print("Original text sample:")
print(job_postings['pst_detail'].iloc[0])
print("\nPreprocessed text sample:")
print(job_postings['preprocessed_text'].iloc[0])
```

## 1.6 텍스트 길이 분석

```python
job_postings['text_length'] = job_postings['preprocessed_text'].str.len()

plt.figure(figsize=(10, 6))
sns.histplot(data=job_postings, x='text_length', bins=50)
plt.title('Distribution of Preprocessed Text Length')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.show()

print(f"평균 텍스트 길이: {job_postings['text_length'].mean():.2f}")
print(f"중앙값 텍스트 길이: {job_postings['text_length'].median():.2f}")
```

## 1.7 데이터 저장

```python
job_postings.to_csv('preprocessed_job_postings.csv', index=False)
print("Preprocessed data saved to 'preprocessed_job_postings.csv'")
```

이 노트북에서는 구인공고 데이터를 로드하고, 텍스트 전처리를 수행한 후, 기본적인 탐색적 데이터 분석을 진행했습니다. 다음 단계에서는 이 전처리된 데이터를 사용하여 숙련 키워드를 추출하게 됩니다.
