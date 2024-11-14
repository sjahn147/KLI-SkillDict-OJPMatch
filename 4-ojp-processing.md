# 구인공고 처리

## 4.1. 처리 결과 

구인공고 3년치(2021,2022,2023) 약 천만 건을 대상으로 사업장정보, 직업코드, 숙련코드 매칭 완료

### 최종 데이터셋 (sfile)의 구조

![image](https://github.com/user-attachments/assets/1e7ba922-3ffc-4f62-9700-6875a0f39a94)

### sfile 데이터셋 정보

| **월/연도** | 2021      | 2022       | 2023       | 계          |
| -------- | --------- | ---------- | ---------- | ---------- |
| **1월**   | 406,558   | 535,749    | 793,761    | 1,736,068  |
| **2월**   | 443,944   | 411,196    | 1,050,766  | 1,905,906  |
| **3월**   | 310,182   | 1,313,664  | 971,564    | 2,595,410  |
| **4월**   | 424,864   | 1,265,800  | 955,571    | 2,646,235  |
| **5월**   | 477,042   | 1,158,484  | 997,761    | 2,633,287  |
| **6월**   | 534,215   | 806,328    | 986,710    | 2,327,253  |
| **7월**   | 521,476   | 851,042    | 1,008,908  | 2,381,426  |
| **8월**   | 499,492   | 1,092,680  | 1,032,231  | 2,624,403  |
| **9월**   | 457,998   | 1,032,396  | 906,131    | 2,396,525  |
| **10월**  | 499,219   | 990,950    | 1,002,294  | 2,492,463  |
| **11월**  | 550,663   | 988,439    | 1,037,860  | 2,576,962  |
| **12월**  | 557,829   | 910,939    | 916,146    | 2,384,914  |
| **계**    | 5,683,482 | 11,357,667 | 11,659,703 | 28,700,852 |

## 4.2. 성능 비교

- **최적화 전**: 80,000 건/시간
- **최적화 후**: 142,857 건/시간
- **성능 향상**: **약 1.79배 개선**

## 4.3. 성능 개선 방법

### 4.3.1. 스킬 사전 로드 및 정규식 패턴 컴파일

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

목적 : 스킬 사전을 로드하고 각 스킬의 **정규식 패턴을 미리 컴파일**.
구현 방법 : 
- Excel 파일에서 **필요한 열만 선택적으로 로드**
- 정규식 패턴 문자열 처리 (raw 문자열 표시 제거, 대소문자 무시 플래그 처리)
- 패턴 컴파일 및 예외 처리

### 4.3.2. 구인공고 텍스트 전처리

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

목적 : 구인공고 **텍스트를 전처리함으로써 잘못된 매칭을 방지**함. 
구현 방법 : 
- **불필요한 정보 제거 (회사 소개, 지원 방법 등)**
- **특정 키워드 이후의 텍스트 제거 (복지 정보, 근무 조건 등)**
- 공백 정규화
### 4.3.3. 행 단위 처리

```python
def process_row(row, skill_dict):
    preprocessed_text = preprocess_job_desc(row['pst_detail'])
    skill_codes = match_skills(preprocessed_text, skill_dict)
    row['skill_code'] = ','.join(skill_codes.keys())
    return row

def process_chunk(chunk, skill_dict):
    return [process_row(row, skill_dict) for _, row in chunk.iterrows()]
```

**청크 단위 처리를 적용하여 메모리 부담을 낮춤**.

### 4.3.4. 파일 처리 및 병렬화

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
구현 방법 : **멀티프로세싱을 통한 병렬 처리** (터미널에서 실행)

### 4.3.5. 로깅 및 예외처리

#### 로깅 및 예외 처리

-  모니터링 :  Python의 내장 `logging` 모듈을 사용
-  예외 처리 :  **프로세스의 중단을 방지**하고 오류 정보를 기록
#### 로깅 설정

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
- 타임스탬프, 프로세스 이름, 로그 레벨, 메시지를 포함
- 파일과 콘솔 모두에 로그 출력

#### 주요 로깅 포인트

1. **파일 처리 시작/종료**
   ```python
   logging.info(f"Processing file: {file_path}")
   logging.info(f"Finished processing {file_path}. Total time: {processing_time:.2f} seconds")
   ```

2. **청크 처리 진행 상황**
   ```python
   logging.info(f"Processed {processed_rows}/{total_rows} rows ({processed_rows/total_rows*100:.2f}%)")
   ```

3. **스킬 사전 로딩**
   ```python
   logging.info(f"Loaded {len(skill_dict)} skills")
   ```
4. **subprocess 로깅**
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
#### 예외 처리

1. **전체 프로세스 레벨**
   ```python
   def process_month(args):
       try:
           # 처리 로직
       except Exception as e:
           logging.error(f"Error processing {year}/{p_num}: {str(e)}")
           return False
   ```

2. **파일 처리 레벨**
   ```python
   def process_job_file(file_path, skill_dict):
       try:
           # 파일 처리 로직
       except Exception as e:
           logging.error(f"Error processing {file_path}: {str(e)}")
           return None
   ```

3. **스킬 사전 로딩 시 정규식 오류 처리**
   ```python
   try:
       skill_dict[row['skill_code']] = {
           'pattern': re.compile(pattern, flags),
           'skill_name': row['skill_name']
       }
   except re.error:
       logging.error(f"Invalid regex pattern for skill_code {row['skill_code']}: {pattern}")
   ```
#### 처리 과정 안정화

1. **치명적이지 않은 오류 무시**
   - 개별 파일 또는 행 처리 실패 시 전체 프로세스 중단 방지
   - **오류 발생 시 해당 항목만 건너뛰고 계속 진행**

2. **상세한 오류 정보 기록**
   - 오류 발생 위치, 관련 데이터, 스택 트레이스 등 상세 정보 로깅

3. **진행 상황 실시간 모니터링**
   - 정기적인 진행 상황 로깅
   - 성능 병목 지점 식별 및 최적화에 활용

4. **재시도 메커니즘**
   - 일시적 오류에 대비한 재시도 로직 구현
   ```python
   @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
   def process_with_retry(func, *args, **kwargs):
       return func(*args, **kwargs)
   ```
