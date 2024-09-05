## 5.3 숙련 코드 생성 함수 (계속)

```python
def generate_skill_code(row):
    category_code = row['category_code']
    subcategory_code = row['subcategory_code']
    skill_number = row['skill_number']
    return f"{category_code}{subcategory_code:03d}{skill_number:04d}"

# 카테고리와 서브카테고리에 코드 할당
df['category_code'] = pd.Categorical(df['category']).codes
df['subcategory_code'] = df.groupby('category')['subcategory'].transform(lambda x: pd.Categorical(x).codes)

# 각 클러스터 내에서 스킬 번호 할당
df['skill_number'] = df.groupby(['category', 'subcategory', 'cluster']).cumcount()

# 숙련 코드 생성
df['skill_code'] = df.apply(generate_skill_code, axis=1)

print("Sample skill codes:")
print(df[['category', 'subcategory', 'keyword', 'skill_code']].head())
```

## 5.4 유사 숙련 통합

```python
def find_similar_skills(group):
    skills = group['keyword'].tolist()
    similar_groups = []
    
    for i, skill1 in enumerate(skills):
        if i in [item for sublist in similar_groups for item in sublist]:
            continue
        current_group = [i]
        for j, skill2 in enumerate(skills[i+1:], start=i+1):
            if fuzz.ratio(skill1, skill2) > 80:  # 80% 이상 유사도를 가진 스킬을 그룹화
                current_group.append(j)
        if len(current_group) > 1:
            similar_groups.append(current_group)
    
    return similar_groups

# 클러스터 내에서 유사한 스킬 그룹 찾기
tqdm.pandas()
df['similar_group'] = df.groupby(['category', 'subcategory', 'cluster'])['keyword'].transform(find_similar_skills)

# 유사 스킬 그룹의 대표 스킬 선택 (가장 긴 키워드를 대표로 선택)
df['representative_skill'] = df.apply(lambda row: max([df.loc[i, 'keyword'] for i in row['similar_group']], key=len) if isinstance(row['similar_group'], list) else row['keyword'], axis=1)

print("\nSample of similar skill groups and their representatives:")
print(df[['keyword', 'similar_group', 'representative_skill']].head(10))
```

## 5.5 최종 숙련 사전 생성

```python
final_skill_dict = df.groupby('skill_code').agg({
    'category': 'first',
    'subcategory': 'first',
    'representative_skill': 'first',
    'keyword': lambda x: ', '.join(set(x)),
    'definition': 'first'
}).reset_index()

print("\nFinal skill dictionary sample:")
print(final_skill_dict.head())

# 최종 숙련 사전 저장
final_skill_dict.to_csv('final_skill_dictionary.csv', index=False)
print("Saved final skill dictionary")
```

이 노트북에서는 클러스터링된 숙련에 대해 고유한 코드를 생성하고, 유사한 숙련을 통합하여 최종 숙련 사전을 만들었습니다. 다음 단계에서는 이 숙련 사전을 사용하여 구인공고와 매칭하는 시스템을 구현하게 됩니다.
