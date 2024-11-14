## 3. 구인공고 매칭
### 3.1. 매칭 방법 검토

#### 3.1.1. 검색 증강 생성 (RAG-LLM)
  - 설명: 이미 존재하는 DB를 참조하여 관련도가 높은 기존의 대안 중에서 LLM이 응답을 선택하도록 하는 방법.
  - 장점: 
	  - 대규모 언어모델의 유연성과 문맥 이해 능력을 활용
	  - 가짜 응답 (Hallucination)을 크게 줄일 수 있음
  - 단점: 
	  - 상용 LLM API를 이용하는 경우 : 높은 API 비용 부담
	  - 로컬 LLM을 이용하는 경우 확인된 문제점 :
		  - 지나치게 높은 연산 부담
		  - 느린 매칭 속도
		  - 낮은 매칭 성능
#### 3.1.2. 지도 학습 (XLM-ROBERTa)
  - 설명: 구인 공고를 읽고 숙련 코드를 예측하는 딥러닝 모델을 훈련.
  - 장점: 
	  - 키워드 검색 방식보다 문맥 유연성이 높음
	  - LLM 방식보다 낮은 연산 부담
  - 단점:
	  - 숙련 사전이 처음부터  훈련 데이터로 만들어진 것이 아니기 때문에 지도 학습 불가. 데이터 사이즈가 너무 작고 품질 낮음. 
	  - 훈련 결과 성능 높이는 노력에 비해 개선 한계 뚜렷할 것으로 판단.
#### 3.2.3. 키워드 기반 매칭
  - 설명: 
	  - 구인공고에서 발견되는 특정 문자열(key)을 기준으로 미리 정제된 숙련 용어(skill)을 매칭
	  - 실제 구인 공고에서 발견되는 특정 문자열을 정제된 숙련 용어와 연결하여 매핑한 사전 자료 필요.
  - 장점: 
	  - 본 연구에서 중요한 **속도, 일관성, 리소스 효율성 면에서 장점**.
  - 단점: 
	  - **문맥 이해 능력과 유연성이 없음.** 
	  - 단, 아래의 시도를 통해 개선될 것으로 기대함.
		  1) 매칭 반응성 개선 :
			  1) 매칭 키를 일정한 규칙에 따라 확장 
			  2) 정규표현식 활용 
		  2) 매칭 정확도 개선 : 유사도 분석 통합
### 3.2. 실제 매칭 방법

#### 3.2.1. 숙련 사전 가공하기
- 키워드 기반 매칭 정교화 : **키워드 확장 및 정규식을 활용한 반응성 개선**
  - 예) 절곡 -> 절곡, 벤딩, bending, ….
  - 공백이나 조사, 단어끼리 붙는 현상 (예 : 열정이 있으신분영어 수업 가능자) 등에 대응할 수 있도록 정규식을 활용하여 검색 방식의 유연성을 추가
  - 6,542 건을 직접 이렇게 가공할 수는 없으므로, Anthropic API에서 Haiku 모델을 사용해 숙련 사전을 정제하였음. 필요 예산 $3로 가공을 완료함.

##### <확장키 생성 프롬프트>
```python

def generate_ext_key(row):
    system_message = r"""당신은 숙련 사전의 한국어 키워드를 기반으로 정규 표현식을 생성하는 전문가입니다.
오직 정규 표현식 패턴만을 생성하고, 추가 설명이나 텍스트 없이 반환하세요.
1. 기본 구조: r'(?i)\b(...)\b'
2. 키워드 확장 규칙:
a) 주어진 모든 키워드(keyword, representative_keyword)를 포함
b) 동의어 및 유사 표현 포함
c) 영어 표현 추가 (가능한 경우)
d) 약어 및 축약형 포함
e) 관련 기술이나 도구 포함 (해당되는 경우)
f) 일반적인 표현과 전문 용어 모두 포함
g) 맥락적 변형 포함
h) 업계 특화 용어 추가
i) 오타 및 일반적인 철자 오류 고려
j) 복합 키워드 분해
k) 숫자 및 단위의 다양한 표현 (해당되는 경우)
l) 동사의 다양한 활용형 포함
m) 카테고리와 서브카테고리 정보를 활용한 관련 용어 포함
3. 정규 표현식 기법:
a) 대소문자 무시: (?i) 사용
b) 단어 경계 설정: \b 사용
c) 선택적 공백: \s? 사용
d) 유사 표현 그룹화: (표현1|표현2|표현3) 형식 사용
4. 특별 고려사항:
a) 카테고리, 서브카테고리 정보를 활용하여 관련 용어 추가
b) 가능한 한 포괄적이고 유연한 매칭이 가능하도록 설계
c) 최신 기술 및 트렌드 반영
예시:
r'(?i)\b(절곡|bending|굽힘|절곡\s?가공|금속\s?절곡|판금\s?절곡|sheet\s?metal\s?bending|절곡\s?기술|벤딩|폴딩|금속\s?성형|판재\s?성형|곡면\s?가공|곡률\s?가공)\b'
r'(?i)\b(절단|cutting|커팅|재단|자르기|절삭|재료\s?절단|금속\s?절단|레이저\s?절단|플라즈마\s?절단|워터젯\s?절단|기계\s?절단|정밀\s?절단|자동\s?절단)\b'
정규 표현식은 반드시 r'로 시작하고 '로 끝나야 합니다."""

    user_message = f"""
다음 스킬을 표현하는 정규 표현식을 작성해주세요. 정규 표현식 패턴만 제공하고,
다른 어떤 설명이나 텍스트도 포함하지 마세요.
스킬 코드: {row['skill_code']}
대분류: {row['category']}
하위분류: {row['subcategory']}
대표 키워드: {row['representative_keyword']}
키워드: {row['keyword']}
"""

    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=500,
            temperature=0.2,
            system=system_message,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

```

![Pasted image 20240907212451](https://github.com/user-attachments/assets/7a05650e-316d-4cb8-8529-09fa978d23bd)

### 3.3. 매칭 로직

#### 3.3.1. 구인 공고 가공
- 추출 시 구인 공고에서 숙련 무관 내용을 절사, 부분 제거한 것과 같은 방식으로, 숙련 사전과 대조하는 숙련 공고 pst_detail 열을 전처리함.
- 전처리하지 않은 구인 공고를 바탕으로 숙련을 찾을 경우 '면접', '채용' 등의 HR 관련 숙련들이 대거 매칭됨.
- 단, 구인 공고 데이터가 10,000~200,000 건 정도로 청크 자체가 크기 때문에 미리 전처리를 하지 않고 숙련 매칭 로직에 행 단위 전처리를 메서드로 포함시킴.
#### #### 3.3.2. 정규식 패턴 검색
  - 정규식 패턴과 일치하는 모든 숙련 코드를 불러옴. pandas와 re 라이브러리로 처리. (API 필요 없음)
  - 테스트 결과, 정규식 패턴 검색 방식의 반응성이 매우 높으므로 아주 조금의 관련성만 있더라도 구인 공고에 붙게 되어 숙련 코드가 수십 개씩 매칭됨. 아예 안 붙는 것보다 나은 결과지만, 추가적인 정제 필요
#### 3.3.3. 관련성 낮은 단어 제거
  - 정규식 패턴과 일치하는 매칭 후보 단어와 실제 구인 공고 pst_detai의 유사도를 계산하고, 유사도 점수가 높은 순으로 정렬함.
  - 이 중에서 점수가 높은 단어만 남기고, 낮은 단어들은 버림. (max =20)
  - 아주 간단한 유사도 계산이므로 처리 속도에는 큰 영향 없음. rapidfuzz를 쓰면 좀 더 빠름.

![Pasted image 20240907213415](https://github.com/user-attachments/assets/805dae94-71a5-4d3e-998a-eb226c81fd86)

![Pasted image 20240907213436](https://github.com/user-attachments/assets/45b3d7e5-16e8-44d1-8c0c-61b8eaf4331e)


```python
#정규식 스킬 사전 불러오기
def load_combined_skill_dict(skill_dict_dir):
    logging.info("Loading and combining skill dictionaries")
    combined_skill_dict = {}
    for file in os.listdir(skill_dict_dir):
        if file.endswith('.xlsx'):
            file_path = os.path.join(skill_dict_dir, file)
            df = pd.read_excel(file_path, header=0)
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
                        combined_skill_dict[row['skill_code']] = (re.compile(pattern, flags), row['representative_keyword'])
                    except re.error:
                        logging.error(f"Invalid regex pattern for skill_code {row['skill_code']}: {pattern}")
    logging.info(f"Loaded {len(combined_skill_dict)} skills in total")
    return combined_skill_dict

def match_skills_with_relevance(text, skill_dict, top_n=20, min_similarity=70):
    if pd.isna(text) or not isinstance(text, str) or text == "":
        return {}, {}

    matched_skills = {} #정규식 패턴과 일치하는 단어들을 set에 담아 중복 제거
    matched_keywords = {}
    for skill_code, (pattern, keyword) in skill_dict.items():
        if pattern.search(text):
            if not isinstance(keyword, str) or len(keyword) < 2:
                continue
            try:
                similarity = fuzz.partial_ratio(keyword.lower(), text.lower())
                if similarity >= min_similarity:
                    matched_skills[skill_code] = similarity
                    matched_keywords[skill_code] = keyword
            except TypeError:
                logging.warning(f"TypeError occurred when processing keyword: {keyword}")
                continue
    
    if len(matched_skills) > top_n: #부분 문자열 유사도를 계산해 상위 20개만 남김김
        top_skills = sorted(matched_skills.items(), key=lambda x: x[1], reverse=True)[:top_n]
        matched_skills = dict(top_skills)
        matched_keywords = {k: matched_keywords[k] for k in matched_skills}
    
    return matched_skills, matched_keywords

def process_job_files(job_files, combined_skill_dict): 
    for job_file in job_files:
        logging.info(f"Starting to process job file: {job_file}")
        try:
            job_df = pd.read_excel(os.path.join(job_dir, job_file)) #구인공고 읽기
        except Exception as e:
            logging.error(f"Error reading job file {job_file}: {str(e)}")
            continue
        
        total_rows = len(job_df)
        logging.info(f"Loaded {total_rows} rows from {job_file}")
        
        skill_code_dict = defaultdict(set)
        skill_name_dict = defaultdict(set)
        
        start_time = time.time()
        for idx, row in tqdm(job_df.iterrows(), total=total_rows, desc=f"Processing {job_file}"):
            try:
                # 전처리 수행 (불필요한 텍스트 잘라내기)
                preprocessed_text = preprocess_job_desc(row['pst_detail'])
                if preprocessed_text == "":
                    logging.warning(f"Empty preprocessed text for row {idx} in {job_file}")
                    continue
                
                # 전처리된 텍스트로 스킬 매칭
                new_skills, new_keywords = match_skills_with_relevance(preprocessed_text, combined_skill_dict, top_n=20, min_similarity=70) #매칭단어 후보들과 해당 구인공고 행의 유사도 계산
                skill_code_dict[idx].update(new_skills.keys())
                skill_name_dict[idx].update(new_keywords.values())
            
            except Exception as e:
                logging.warning(f"Error processing row {idx} in {job_file}: {str(e)}")
                continue
            
            if idx % 100 == 0 or idx == total_rows - 1:
                progress = (idx + 1) / total_rows * 100
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (idx + 1) * total_rows
                remaining_time = estimated_total_time - elapsed_time
                logging.info(f"Processing {os.path.basename(job_file)}: "
                             f"{idx+1}/{total_rows} rows ({progress:.2f}%) | "
                             f"Elapsed: {elapsed_time:.2f}s | "
                             f"Estimated remaining: {remaining_time:.2f}s")
        
        job_df['skill_code'] = [','.join(skill_code_dict[idx]) for idx in range(len(job_df))]
        job_df['skill_name'] = [','.join(skill_name_dict[idx]) for idx in range(len(job_df))] # 쉼표로 구분하여 열에 join
        
        final_output_path = os.path.join(results_dir, f"final_{job_file}")
        job_df.to_excel(final_output_path, index=False)
        logging.info(f"Final results saved to {final_output_path}")
```

![Pasted image 20240907213720](https://github.com/user-attachments/assets/83c4f2a0-130c-4cf1-b652-27a6ee04df80)
