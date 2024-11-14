# 한국숙련분류체계 구축 및 구인공고 매칭 프로젝트

## 프로젝트 개요

### 기술 스택
- **Languages & Environments**: Python 3.x, Jupyter Notebook, Google Colab, conda
- **Core Libraries**: pandas, numpy, NLTK, spaCy
- **Text Analysis**: Sentence-BERT, FuzzyMatching
- **Machine Learning**: XLM-ROBERTa, DBSCAN
- **Process Management**: multiprocessing, subprocess, logging
- **Infrastructure**: NAS, Elasticsearch
- **Database & Visualization**: PostgreSQL, Tableau
- **API Integration**: Anthropic Claude API

### 목적 

- 국내 최초 온라인 구인공고 기반 숙련분류체계 구축 및 기업 자료 연계
- 해외 ESCO, LightCast와 같은 숙련분류체계를 한국 실정에 맞게 개발  

### 분석 범위

- 2016년 - 2024년 국내 온라인 구인공고 데이터(OJP)
	```사람인, 잡코리아, 워크넷, 인크루트, 잡플래닛, 인디드, 원티드, 커리어, 캐치, 비즈니스피플, 피플앤잡 등 모든 구인구직 사이트에서 누락 없이 수집하여 포괄성(Comprehensiveness) 기준을 충족함```
- 전년도 과제 IT 직종 대상에서 금년 과제는 전체 업종 대상으로 확대

### 분석 배경 
- 숙련(skill)은 근로자의 직무 수행 능력을 의미하며, 노동시장 분석의 핵심 요소.
- 실제 온라인 구인 공고에서 추출한 숙련 수요 정보를 기업 자료와 연계하면 다양한 분야에서 활용이 가능.
- 해외에서는 숙련분류체계가 이미 마련되어 공공 영역 및 민간 기업에서 활용 중
  - e.g. ESCO(European Skills, Competences, and Occupations) : 개념적 숙련분류체계
  - e.g. LightCast: 실제 온라인구인공고를 분석하여 구축한 경험적 숙련분류체계
- 그러나 한국에는 아직 이러한 독자적인 숙련분류체계가 부재.
- 2024년도 프로젝트는 국내에서 2016~2024년 기간 동안 수집한 온라인 구인공고로부터 국내 최초로 숙련분류체계를 구축하고, 이를 한국평가데이터(KoDATA) 등 실제 기업 데이터와 연계하려는 목적을 갖고 있음. 

## 프로젝트 수행 단계별 역할

<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
    <tr>
        <th style="width: 20%; border: 1px solid #000; padding: 8px; background-color: #f5f5f5;">작업단계</th>
        <th style="width: 60%; border: 1px solid #000; padding: 8px; background-color: #f5f5f5;">단계 상세</th>
        <th style="width: 20%; border: 1px solid #000; padding: 8px; background-color: #f5f5f5;">담당</th>
    </tr>
    <tr>
        <td style="border: 1px solid #000; padding: 8px;">1. 데이터수집</td>
        <td style="border: 1px solid #000; padding: 8px;">
            1-1. 원자료 Web Scraping (2016-2024년 전국 모든 온라인구인공고)<br>
            1-2. 중복 삭제, 날짜별 정렬 (몽태컴퍼니)
        </td>
        <td style="border: 1px solid #000; padding: 8px;">Web Scraping팀</td>
    </tr>
    <tr>
        <td style="border: 1px solid #000; padding: 8px;">2. 숙련사전구축</td>
        <td style="border: 1px solid #000; padding: 8px;">
            2-1. 구인공고 샘플링 및 숙련 용어 추출: 업종 분포 고려, API 활용<br>
            2-2. 숙련용어정제: 중복 단어 제거, 정규식 활용 불용어, 공백, 특수문자 제거<br>
            2-3. 숙련용어통합 및 분류<br>
            &nbsp;&nbsp;&nbsp;① 숙련용어분류: LLM을 활용한 초벌 분류 및 소프트 스킬 식별을 위한 질적 코딩<br>
            &nbsp;&nbsp;&nbsp;② 군집분석: 숙련 용어마다 용어의 정의 생성하여 벡터화, 클러스터 생성<br>
            &nbsp;&nbsp;&nbsp;③ 용어정제: 클러스터 정보를 활용한 퍼지 매칭
        </td>
        <td style="border: 1px solid #000; padding: 8px;">전병유·안성준<br>(한신대 산학협력단)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #000; padding: 8px;">3. 구인공고매칭</td>
        <td style="border: 1px solid #000; padding: 8px;">
            3-1. 산업, 직업 코드 부여: KoBERT 활용하여 개발한 자동직업분류모델(XLM-RoBERTa)<br>
            3-2. 숙련 코드 부여: RAG-LLM, 키워드 기반 매칭 등 대안적 매칭 방법 추가 탐색
        </td>
        <td style="border: 1px solid #000; padding: 8px;">전병유·안성준<br>(한신대 산학협력단)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #000; padding: 8px;">4. 기업 자료 연계</td>
        <td style="border: 1px solid #000; padding: 8px;">4-1. 구인공고에 사업자등록번호를 매칭하기 위한 연계키 준비</td>
        <td style="border: 1px solid #000; padding: 8px;">몽태컴퍼니<br>안성준</td>
    </tr>
    <tr>
        <td style="border: 1px solid #000; padding: 8px;">5. 최종 파일 준비</td>
        <td style="border: 1px solid #000; padding: 8px; ">
            5-1. 구인공고별 정렬(Postings 파일)<br>
            5-2. 숙련별 정렬(Skills 파일)
        <td style="border: 1px solid #000; padding: 8px;">전병유·안성준<br>몽태컴퍼니</td>   
    </tr>
    <tr>
        <td style="border: 1px solid #000; padding: 8px;">6. 데이터 활용 연구</td>
        <td style="border: 1px solid #000; padding: 8px;">6-1. AI Index 생성 등: AI Index 생성에 필요한 핵심 AI 숙련 키워드 목록과 AI 인접 숙련 키워드 목록 선정</td>
        <td style="border: 1px solid #000; padding: 8px;">장지연 외</td>       
    </tr>
</table>
<p style="text-align: right; font-size: 0.9em;">※굵은 테두리 영역: 담당 / 회색 영역: 협업</p>

## 프로젝트 주요 성과
### **한국형 숙련분류체계 신규 구축 완료**  

약 33,000개의 초기 숙련 키워드에서 7,192개의 정제된 숙련 사전 구축

#### 도메인별 주요 스킬 키워드
![image](https://github.com/user-attachments/assets/84c9ab1b-374a-4d4e-8146-bd771125b5ac)
![image](https://github.com/user-attachments/assets/c7a4ce7a-a613-4854-964c-6e175858ed96)

### 온라인 구인공고 데이터셋 신규 구축 완료 

2021-2023 구인공고 데이터 약 1,000만 건의 처리 및 사업체 정보, 직업, 업종, 숙련코드 매칭 진행

#### 온라인 구인공고 데이터셋 처리 결과

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

#### KECO 세분류 직종(450개)의 숙련 포트폴리오 파악  

![Pasted image 20241114150900](https://github.com/user-attachments/assets/6ac49ef9-e2d8-430f-abd5-67356736ebe4)

![image](https://github.com/user-attachments/assets/e186978a-4374-4079-8f48-49ce3c3897d6)

### 데이터 활용 연구

#### AI 도입 비율 분석

한국 데이터를 통한 OECD 등 해외 연구보고서 결과 재현 성공 (Acemoglu et al. 2022; Calvino, F., & Fontanelli, L. 2023)

#### 2021-2023 연월별 인공지능 숙련 키워드 건수

인공지능 숙련에 대한 수요는 꾸준히 늘고 있으며, 국내 기업의 AI 도입률은 **2021년 2.9%에서 2021년 4.0% 로 증가 중**. 

   ![2021-2023 연월별 인공지능 숙련 키워드 건수](https://github.com/user-attachments/assets/631fa500-5260-40a3-854e-6adff9f189ba)

#### 기업 규모별 AI 도입 비율

**AI 고용은 주로 1,000인 이상 대기업에 의해 주도**되는 경향

![Pasted image 20241114150227](https://github.com/user-attachments/assets/53219a0a-507c-4df1-bf74-9875b44f1670)

#### 업종별 AI 도입 비율
   
   - 업종별 AI 도입률은 **정보통신업 > 금융업 >전문서비스업> 장비 제조업> 도소매업> 부동산업 > 기타 제조업 > 서비스업 > 음식숙박업> 사회간접> 사회서비스업 > 운수창고업 > 농업** 순  

#### AI 도입비율과 AI 노출도의 관계

**AI 도입률이 높은 기업은 AI 노출도 역시 유의하게 높음**. 기존에 관계가 뚜렷하지 않았던 AI 노출도의 AI 실제 도입률의 실제 상관관계 확인 
![image](https://github.com/user-attachments/assets/a51d9dff-b082-4d3c-a134-037a8908012f)
## 도전 과제 및 해결 방안

1. **다양한 표현의 숙련 용어 통합**
   - 문제 :
      - 33,000개 초기 키워드에서 숙련 용어 선별 및 통합 필요.
      - 동일 의미의 다양한 표현/ 근무조건, 학력 등 비숙련 단어/ 소프트숙련 식별 필요/ 도메인 혼재 등
      - 자연어처리 능력 및 스킬에 대한 사전지식이 요구되어 기계적 분류의 한계
   - 해결 :
      - LLM으로 비숙련 단어 및 소프트 스킬 분리, 도메인 초벌 분류 수행
      - 식별된 도메인별로 기계적 클러스터링을 통해 유사 표현 통합 -> 7,192개의 단어로 정제 통합
3. **구인공고 매칭 시스템 구축**
   - 문제 :
      - 구인공고 텍스트를 읽고 숙련 코드를 매칭해야함.
      - 1,000만 건 구인공고 데이터의 효율적 처리 필요
      - ROBERTa나 RAG-LLM을 시도했으나 낮은 성능과 속도
   - 해결 :
      - 복잡한 모델링 대신 pandas와 Regex, 유사도 분석 등 간단한 로직의 조합으로 빠르고 정확한 매칭 구현
      - 데이터 최적화 및 병렬 처리 구현으로 처리 속도의 추가적인 개선  (평균 처리 속도: 14만 건/시간)

## 프로젝트 구조도

### 전체 프로젝트 구조
<img src="https://github.com/user-attachments/assets/0de3721f-f226-4113-b7a0-baa704c8db6e" width="100%" alt="프로젝트 구조 1">

#### 1. 숙련사전 구축 프로세스
<img src="https://github.com/user-attachments/assets/c3db8244-3691-4f0e-a06f-8322c3e5c4f1" width="80%" alt="프로젝트 구조 2">

#### 2. 숙련코드 매칭 프로세스

<img src="https://github.com/user-attachments/assets/52025d1a-e8a4-47a3-ae7d-72435ecb4c67" width="80%" alt="프로젝트 구조 4">


## 주요 기능 및 구현 방법

### 숙련 사전 구축
#### 구인공고 전처리 및 숙련 키워드 추출
   - 구인공고 섹션 분할을 통한 핵심 내용 추출
   - LLM을 활용한 구인공고 텍스트에서 숙련 관련 키워드 추출
#### 숙련 키워드 통합
   - LLM을 활용한 초기 숙련 분류
   - DBSCAN 알고리즘을 이용한 유사 숙련 클러스터링 및 숙련 코드 생성
   - 클러스터별 퍼지 매칭을 통한 자동 레이블링     
#### 숙련 사전 구축
   - 정규표현식 기반의 키워드 확장을 포함한 숙련사전 정제

### 구인공고 데이터 처리

#### 숙련코드 2단계 매칭 프로세스 
- **1차 매칭: 광범위한 후보군 추출**
   - 정규표현식 패턴 기반의 초기 매칭 수행
   - 불필요한 채용공고 텍스트 제거로 노이즈 감소
   - 숙련 키워드별 정규식 패턴으로 잠재적 매칭 단어 최대 추출
- **2차 매칭: 유사도 기반 정밀 선별**
   - rapidfuzz 라이브러리의 partial_ratio 알고리즘 활용
   - 1차 매칭된 단어들 중 70% 이상 유사도를 가진 항목 필터링
   - 유사도 점수 기준 상위 스킬 코드 최종 선정

#### 대규모 데이터 처리 아키텍처
- **데이터 파일 구조**
    - **pfile (구인공고 상세 데이터)**
        - 구인공고 원문, 급여, 지역 등 상세 정보
        - 숙련코드, 직업코드, 사업자등록번호 연계
    - **sfile (숙련 데이터셋)**
        - 구인공고별 4~20개 숙련 단어 전개
        - 숙련 분류체계 정보 포함 (대/중/소분류, 소프트 숙련 여부)
        - 약 2,800만 건의 레코드
- **데이터 처리 파이프라인**
    - **데이터 생성 단계**
        - Process Pool 기반 12코어 병렬 처리
        - 연도/월별 폴더 단위 작업 분배
        - subprocess를 통한 워커 프로세스 실행
    - **데이터 구조 최적화**
        - DataFrame 청크 처리로 메모리 사용량 제어
        - Dictionary 기반 O(1) 패턴 매칭
        - string array 기반 유사도 계산
        - List 구조의 배치 처리
	- **모니터링 및 자동화**
	    - 처리 진행률 실시간 모니터링
	    - 프로세스별 처리 시간 측정
	    - 에러 핸들링 및 로깅
	    - 연도/월별 자동 순회 처리
	- **데이터 저장 및 분석**
		- NAS 서버에 결과 파일 저장
		- PostgreSQL에서 pfile-sfile 연계 처리
		- Tableau 연동을 통한 시각화
   
## 향후 개선 목표

### 데이터 인프라 및 파이프라인 고도화
- **Hadoop 데이터 레이크 구축**
    - 원본 구인공고 데이터 적재 및 이력 관리
- **Spark 기반 처리 시스템 구축**
    - 기존 Python 처리 로직 마이그레이션 (Spark SQL, Spark UDF 전환)
    - 신규 키워드 자동 감지 및 분류
    - 기존 사전과의 중복 검사 로직 구현
- **PostgreSQL 분석 환경 구축**
    - 정제 데이터 적재 및 버전 관리
    - Tableau 연동 유지
- **Airflow 기반 파이프라인 자동화**
    - 데이터 수집-처리-적재 프로세스 자동화
    - 처리 단계별 검증 포인트 설정
    - 작업 실패 대응 및 모니터링
- 품**질 관리 프로세스 수립**
    - 신규 추가 키워드의 주기적 검토 체계
    - 업데이트 결과의 영향도 분석 시스템
### 숙련 매칭 고도화
- **RAG-LLM 성능 개선 시도**
- **도메인 특화 딥러닝 모델 개발**
    - 구축된 1,000만 건의 구인공고-숙련코드 매칭 데이터 활용
    - 직업분류 및 업종코드와의 교차 검증을 통한 고품질 훈련 데이터셋 구축
    - 검증된 데이터셋 기반의 모델 학습
## 참조

- 장지연(2024) 직종별 AI 노출도. 한국노동연구원. 노동리뷰 2024년5월호.
- 전병유, 정준호, & 장지연. (2022). 인공지능 (AI) 의 고용과 임금 효과. 경제연구, 40(1), 133-156.
- Acemoglu, D., Autor, D., Hazell, J., & Restrepo, P. (2022). Artificial intelligence and jobs: Evidence from online vacancies. Journal of Labor Economics, 40(S1), S293-S340.
- Babina, T., Fedyk, A., He, A. X., & Hodson, J. (2020). Artificial Intelligence, Firm Growth, and Industry Concentration. SSRN Scholarly Paper ID 3651052. Social Science Research Network, Rochester, NY.
- Calvino, F. and L. Fontanelli (2023), “A portrait of AI adopters across countries: Firm characteristics, assets’ complementarities and productivity”, OECD Science, Technology and Industry Working Papers, No. 2023/02, OECD Publishing, Paris, https://doi.org/10.1787/0fb79bb9-en.
- Felten, E. W., Raj, M., & Seamans, R. (2018, May). A method to link advances in artificial intelligence to occupational abilities. In AEA Papers and Proceedings (Vol. 108, pp. 54-57).
- Felten, E. W., Raj, M., & Seamans, R. (2023). Occupational heterogeneity in exposure to generative ai. Available at SSRN 4414065. 
- Gmyrek, P., Berg, J., & Bescond, D. (2023). Generative AI and jobs: A global analysis of potential effects on job quantity and quality. ILO Working Paper, 96.
