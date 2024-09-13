# 구인공고 스킬 추출 프로젝트 상세 코드 설명 및 성능 비교

## 성능 비교

- 최적화 전: 80,000 건/시간
- 최적화 후: 142,857 건/시간
- 성능 향상: 약 1.79배 개선

## 1. 스킬 사전 로드 및 정규식 패턴 컴파일

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
```

목적 : 스킬 사전을 로드하고 각 스킬의 정규식 패턴을 미리 컴파일.
구현 방법 : 
- Excel 파일에서 필요한 열만 선택적으로 로드
- 정규식 패턴 문자열 처리 (raw 문자열 표시 제거, 대소문자 무시 플래그 처리)
- 패턴 컴파일 및 예외 처리

## 2. 구인공고 텍스트 전처리

```python
def preprocess_job_desc(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    patterns_to_remove = [
        r"기업\s*소개\s*또는\s*채용\s*안내를\s*작성해\s*보세요.*",
        r"\*{3}\s*입사\s*지원",
        # ... (다른 패턴들)
    ]
    
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    cut_keywords = r'혜택\s*및\s*복지|근무\s*조건|전형\s*절차|...'
    match = re.search(cut_keywords, text, re.IGNORECASE)
    if match:
        text = text[:match.start()]
    
    return re.sub(r'\s+', ' ', text).strip()
```

목적 : 구인공고 텍스트를 전처리함으로써 잘못된 매칭을 방지함. 
구현 방법 : 
- 불필요한 정보 제거 (회사 소개, 지원 방법 등)
- 특정 키워드 이후의 텍스트 제거 (복지 정보, 근무 조건 등)
- 공백 정규화

## 3. 스킬 매칭 및 유사도 계산

```python
def match_skills(text, skill_dict, top_n=20, min_similarity=70):
    matched_skills = {}
    for skill_code, skill_info in skill_dict.items():
        if skill_info['pattern'].search(text):
            similarity = fuzz.partial_ratio(skill_info['skill_name'].lower(), text.lower())
            if similarity >= min_similarity:
                matched_skills[skill_code] = similarity
    
    return dict(sorted(matched_skills.items(), key=lambda x: x[1], reverse=True)[:top_n])
```

목적 : 전처리된 텍스트에서 스킬을 매칭하고 유사도를 계산합니다.
구현 방법 : 
- 정규식 패턴 매칭을 통한 초기 필터링
- 부분 문자열 유사도 계산 (rapidfuzz 라이브러리 사용)
- 최소 유사도 임계값 적용
- 상위 N개 결과만 반환

## 4. 행 단위 처리

```python
def process_row(row, skill_dict):
    preprocessed_text = preprocess_job_desc(row['pst_detail'])
    skill_codes = match_skills(preprocessed_text, skill_dict)
    row['skill_code'] = ','.join(skill_codes.keys())
    return row

def process_chunk(chunk, skill_dict):
    return [process_row(row, skill_dict) for _, row in chunk.iterrows()]
```

청크 단위 처리 구현를 구현하여 메모리 부담을 낮춤

## 5. 파일 처리 및 병렬화

```python
def process_job_file(file_path, skill_dict):
    df = pd.read_excel(file_path)
    chunk_size = 1000
    chunks = [df[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

    results = []
    for i, chunk in enumerate(chunks):
        chunk_result = process_chunk(chunk, skill_dict)
        results.extend(chunk_result)
        
        if (i+1) % 10 == 0:
            processed_rows = (i+1) * chunk_size
            logging.info(f"Processed {processed_rows}/{len(df)} rows ({processed_rows/len(df)*100:.2f}%)")

    return pd.DataFrame(results)

def run_worker(args):
    year, folder_path = args
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
    for file_path in files:
        subprocess.run([python_executable, worker_script, file_path], check=True)

def main():
    # ... (연도 및 폴더 경로 설정)
    with mp.Pool(processes=num_processes) as pool:
        list(pool.imap_unordered(run_worker, all_folder_paths))
```

목적 : 전체 파일 처리 및 병렬화
구현 방법 : 
- 청크 단위 파일 처리로 메모리 사용 최적화
- 멀티프로세싱을 통한 병렬 처리

## 6. 로깅 및 예외처리

### 로깅 및 예외 처리

- 프로젝트의 안정성과 모니터링을 위해 Python의 내장 `logging` 모듈을 사용
- 적절한 예외 처리를 통해 프로세스의 중단을 방지하고 오류 정보를 기록함

### 로깅 설정

```python
import logging

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(processName)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("process.log"),
                        logging.StreamHandler()
                    ])
```
- 로그 레벨을 INFO로 설정하여 주요 진행 상황을 기록
- 타임스탴프, 프로세스 이름, 로그 레벨, 메시지를 포함하는 포맷 사용
- 파일과 콘솔 모두에 로그 출력

### 주요 로깅 포인트

1. 파일 처리 시작/종료
   ```python
   logging.info(f"Processing file: {file_path}")
   logging.info(f"Finished processing {file_path}. Total time: {processing_time:.2f} seconds")
   ```

2. 청크 처리 진행 상황
   ```python
   logging.info(f"Processed {processed_rows}/{total_rows} rows ({processed_rows/total_rows*100:.2f}%)")
   ```

3. 스킬 사전 로딩
   ```python
   logging.info(f"Loaded {len(skill_dict)} skills")
   ```
4. subprocess 로깅
```python
def run_worker(args):
    year, folder_path = args
    try:
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]
        for file_path in files:
            subprocess.run([python_executable, worker_script, file_path], check=True)
        logging.info(f"Completed processing folder {folder_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error processing folder {folder_path}: {str(e)}")
```

### 예외 처리

1. 전체 프로세스 레벨
   ```python
   def process_month(args):
       try:
           # 처리 로직
       except Exception as e:
           logging.error(f"Error processing {year}/{p_num}: {str(e)}")
           return False
   ```

2. 파일 처리 레벨
   ```python
   def process_job_file(file_path, skill_dict):
       try:
           # 파일 처리 로직
       except Exception as e:
           logging.error(f"Error processing {file_path}: {str(e)}")
           return None
   ```

3. 스킬 사전 로딩 시 정규식 오류 처리
   ```python
   try:
       skill_dict[row['skill_code']] = {
           'pattern': re.compile(pattern, flags),
           'skill_name': row['skill_name']
       }
   except re.error:
       logging.error(f"Invalid regex pattern for skill_code {row['skill_code']}: {pattern}")
   ```

### 처리 과정 안정화

1. 치명적이지 않은 오류 무시
   - 개별 파일 또는 행 처리 실패 시 전체 프로세스 중단 방지
   - 오류 발생 시 해당 항목만 건너뛰고 계속 진행

2. 상세한 오류 정보 기록
   - 오류 발생 위치, 관련 데이터, 스택 트레이스 등 상세 정보 로깅
   - 문제 해결 및 디버깅 용이성 향상

3. 진행 상황 실시간 모니터링
   - 정기적인 진행 상황 로깅으로 장기 실행 프로세스 모니터링 가능
   - 성능 병목 지점 식별 및 최적화에 활용

4. 재시도 메커니즘
   - 일시적 오류(네트워크 문제 등)에 대비한 재시도 로직 구현
   ```python
   @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
   def process_with_retry(func, *args, **kwargs):
       return func(*args, **kwargs)
   ```
