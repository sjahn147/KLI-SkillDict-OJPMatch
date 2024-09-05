# 5. 숙련 코드 생성 및 통합

이 노트북에서는 클러스터링된 숙련에 대해 코드를 생성하고 유사 숙련을 통합하는 과정을 다룹니다.

## 5.1 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import fuzz
```

## 5.2 클러스터링된 숙련 데이터 로드

```python
df = pd.read_csv('clustered_skills.csv')
print(f"Loaded {len(df)} clustered skills")
```

## 5.3 숙련 코드 생성 함수

```python
def generate_skill_code(row):
    category_code = row['category_code']
    subcategory_