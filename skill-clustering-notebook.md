# 4. 숙련 클러스터링

이 노트북에서는 분류된 숙련 키워드를 클러스터링하는 과정을 다룹니다.

## 4.1 필요한 라이브러리 임포트

```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import matplotlib.pyplot as plt
```

## 4.2 분류된 숙련 데이터 로드

```python
df = pd.read_csv('classified_skills.csv')
print(f"Loaded {len(df)} classified skills")
```

## 4.3 텍스트 임베딩 (SBERT)

```python
# SBERT 모델 로드
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# 임베딩 함수 정의
def get_embedding(text, model):
    return model.encode(text)

# 키워드와 정의를 결합하여 임베딩
tqdm.pandas()
df['combined_text'] = df['keyword'] + " " + df['definition']
df['embedding'] = df['combined_text'].progress_apply(lambda x: get_embedding(x, model))
```

## 4.4 DBSCAN 클러스터링

```python
# DBSCAN 클러스터링 수행
eps = 0.3  # 이 값은 실험을 통해 조정될 수 있습니다
min_samples = 2

dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
df['cluster'] = dbscan.fit_predict(np.stack(df['embedding'].values))

print(f"Number of clusters: {len(set(df['cluster'])) - 1}")  # -1은 노이즈 포인트를 나타냅니다
print(f"Number of noise points: {sum(df['cluster'] == -1)}")
```

## 4.5 클러스터링 결과 분석

```python
# 클러스터 크기 분포
cluster_sizes = df[df['cluster'] != -1]['cluster'].value_counts()

plt.figure(figsize=(12, 6))
cluster_sizes.hist(bins=50)
plt.title('Distribution of Cluster Sizes')
plt.xlabel('Cluster Size')
plt.ylabel('Count')
plt.show()

# 가장 큰 클러스터 5개 살펴보기
print("Top 5 largest clusters:")
for cluster in cluster_sizes.nlargest(5).index:
    print(f"\nCluster {cluster}:")
    print(df[df['cluster'] == cluster]['keyword'].values)
```

## 4.6 클러스터 시각화 (t-SNE)

```python
from sklearn.manifold import TSNE

# t-SNE를 사용하여 고차원 임베딩을 2D로 축소
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(np.stack(df['embedding'].values))

# 시각화
plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=df['cluster'], cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('t-SNE visualization of skill clusters')
plt.show()
```

## 4.7 클러스터링 결과 저장

```python
df.to_csv('clustered_skills.csv', index=False)
print("Saved clustered skills")
```

이 노트북에서는 SBERT를 사용하여 숙련 키워드와 정의를 임베딩하고, DBSCAN 알고리즘을 사용하여 클러스터링을 수행했습니다. 또한 클러스터링 결과를 분석하고 시각화했습니다. 다음 단계에서는 이 클러스터링 결과를 바탕으로 숙련 코드를 생성하고 통합하게 됩니다.
