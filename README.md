# 한국형 숙련분류체계 구축 및 구인공고 매칭 프로젝트

## 프로젝트 개요

- 목적: 국내 최초 온라인 구인공고 기반 숙련분류체계 구축 및 기업 자료 연계
- 기간: 2016년 - 2024년 국내 온라인 구인공고 데이터(OJP) 활용 ```사람인, 잡코리아, 워크넷, 인크루트, 잡플래닛, 인디드, 원티드, 커리어, 캐치, 비즈니스피플, 피플앤잡 등 모든 구인구직 사이트에서 누락 없이 수집하여 포괄성(Comprehensiveness) 기준을 충족함```
- 특징: 해외 ESCO, LightCast와 같은 숙련분류체계를 한국 실정에 맞게 개발  대상: 이전 과제 IT 직종 대상에서 전체 업종 대상으로 확대
- 배경 :
   - 숙련(skill)은 근로자의 직무 수행 능력을 의미하며, 노동시장 분석의 핵심 요소.
   - 실제 온라인 구인 공고에서 추출한 숙련 수요 정보를 기업 자료와 연계하면 다양한 분야에서 활용이 가능.
   - 해외에서는 숙련분류체계가 이미 마련되어 공공 영역 및 민간 기업에서 활용 중
      - e.g. ESCO(European Skills, Competences, and Occupations) : 개념적 숙련분류체계
      - e.g. LightCast: 실제 온라인구인공고를 분석하여 구축한 경험적 숙련분류체계
   - 그러나 한국에는 아직 이러한 독자적인 숙련분류체계가 부재.
   - 2024년도 프로젝트는 국내에서 2016~2024년 기간 동안 수집한 온라인 구인공고로부터 국내 최초로 숙련분류체계를 구축하고, 이를 한국평가데이터(KoDATA) 등 실제 기업 데이터와 연계하려는 목적을 갖고 있음. 
   - 2023년도 프로젝트는 IT 업종 대상으로만 진행하였으나, 2024년도 프로젝트에서는 전체 업종으로 대상 확대.

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
        <td style="border: 1px solid #000; padding: 8px; background-color: #333; color: white;">5. 최종 파일 준비</td>
        <td style="border: 1px solid #000; padding: 8px; background-color: #333; color: white;">
            5-1. 구인공고별 정렬(Postings 파일)<br>
            5-2. 숙련별 정렬(Skills 파일)
        <td style="border: 1px solid #000; padding: 8px;">전병유·안성준<br>몽태컴퍼니</td>   
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #000; padding: 8px;">6. 데이터 활용 연구</td>
        <td style="border: 1px solid #000; padding: 8px;">6-1. AI Index 생성 등: AI Index 생성에 필요한 핵심 AI 숙련 키워드 목록과 AI 인접 숙련 키워드 목록 선정</td>
        <td style="border: 1px solid #000; padding: 8px;">장지연 외</td>       
    </tr>
</table>
<p style="text-align: right; font-size: 0.9em;">※굵은 테두리 영역: 담당 / 회색 영역: 협업</p>

## 프로젝트 성과
- **한국형 숙련분류체계 신규 구축 완료** : 약 33,000개의 초기 숙련 키워드에서 7,192개의 정제된 숙련 사전 구축
- **온라인 구인공고 데이터셋 신규 구축 완료** : 2021-2023 구인공고 데이터 약 1,000만 건의 처리 및 사업체 정보, 직업, 업종, 숙련코드 매칭 
- **데이터 활용 연구** : 한국 데이터를 통한 OECD 등 해외 연구보고서 결과 재현 성공 (Acemoglu et al. 2022; Calvino, F., & Fontanelli, L. 2023)
   - 국내 기업 AI 도입률은 3.9% ~ 5.9% 범위에서 증가 중.
   - 국내 데이터로 기존에 관계가 뚜렷하지 않았던 AI 노출도의 AI 실제 도입률의 상관관계 확인 (기업 규모와 무관하게 0.6 수준에서 정적 관계)

## 도전 과제 및 해결 방안
1. 다양한 표현의 숙련 용어 통합
   - 문제 :
      - 33,000개 초기 키워드에서 숙련 용어 선별 및 통합 필요
      - 동일 의미의 다양한 표현/ 근무조건, 학력 등 비숙련 단어/ 소프트숙련 식별 필요/ 도메인 혼재 등
      - 자연어처리 능력 및 스킬에 대한 사전지식이 요구되어 기계적 분류의 한계
   - 해결 :
      - LLM으로 비숙련 단어 및 소프트 스킬 분리, 도메인 초벌 분류 수행
      - 식별된 도메인별로 기계적 클러스터링을 통해 유사 표현 통합 -> 7,192개의 단어로 정제 통합
3. 대규모 구인공고의 매칭 시스템 구축
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

### 1. 숙련사전 구축 프로세스
<img src="https://github.com/user-attachments/assets/c3db8244-3691-4f0e-a06f-8322c3e5c4f1" width="80%" alt="프로젝트 구조 2">

### 2. 숙련코드 매칭 프로세스

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
- 1차 매칭: 광범위한 후보군 추출
   - 정규표현식 패턴 기반의 초기 매칭 수행
   - 불필요한 채용공고 텍스트 제거로 노이즈 감소
   - 숙련 키워드별 정규식 패턴으로 잠재적 매칭 단어 최대 추출
- 2차 매칭: 유사도 기반 정밀 선별
   - rapidfuzz 라이브러리의 partial_ratio 알고리즘 활용
   - 1차 매칭된 단어들 중 70% 이상 유사도를 가진 항목 필터링
   - 유사도 점수 기준 상위 스킬 코드 최종 선정

#### 대규모 데이터 처리 아키텍처

- 멀티프로세싱 기반 병렬 처리
   - Process Pool을 활용한 12개 코어 병렬 처리
   - 연도별/월별 폴더 단위 작업 분배
   - subprocess를 통한 워커 프로세스 실행
- 다중 데이터 구조 활용
   - DataFrame 청크 처리로 멀티프로세싱 메모리 사용량 제어
   - Dictionary 구조: 컴파일된 정규식 패턴과 스킬명을 O(1) 시간에 검색
   - string array 기반 partial_ratio 계산으로 연산 효율화
   - List 구조: 중간 결과 수집 및 배치 처리
- 로깅 및 모니터링
   - 처리 진행률 실시간 모니터링
   - 프로세스별 처리 시간 측정 및 기록
   - 에러 핸들링 및 로그 기록
- 자동화된 파일 처리
  - 연도/월별 디렉토리 구조 기반의 자동 순회 처리
  - 분할된 파일 단위의 일괄 처리
  - NAS 서버 연동을 통한 처리 결과 동기화

## 주요 기술 스택
- 프로그래밍 언어: Python
- 데이터 처리: pandas, numpy
- 자연어 처리: NLTK, spaCy
- 머신러닝: scikit-learn
- 딥러닝: Sentence-BERT (텍스트 임베딩)
- 기타: OpenAI API, Anthropic API
   
## 향후 개선 방향

### 숙련 매칭 고도화
- RAG-LLM 성능 개선
- 도메인 특화 딥러닝 모델 개발
   - 구축된 1,000만 건의 구인공고-숙련코드 매칭 데이터 활용
   - 직업분류 및 업종코드와의 교차 검증을 통한 고품질 훈련 데이터셋 구축
   - 검증된 데이터셋 기반의 모델 학습

### 실시간 구인공고 처리 시스템 구축
- 실시간 처리 파이프라인 구축
- 적합한 데이터베이스 시스템으로 마이그레이션 검토

### 숙련 사전 자동 업데이트 체계 구축
- 증분 업데이트 시스템 개발
   - 신규 키워드 자동 감지 및 분류
   - 기존 사전과의 중복 검사 로직 구현
   - 버전 관리 및 변경 이력 추적 시스템 도입
- 품질 관리 프로세스 수립
   - 자동 검증 규칙 정의
   - 신규 추가 키워드의 주기적 검토 체계
   - 업데이트 결과의 영향도 분석 시스템


## 참조

- 장지연(2024) 직종별 AI 노출도. 한국노동연구원. 노동리뷰 2024년5월호.
- 전병유, 정준호, & 장지연. (2022). 인공지능 (AI) 의 고용과 임금 효과. 경제연구, 40(1), 133-156.
- Acemoglu, D., Autor, D., Hazell, J., & Restrepo, P. (2022). Artificial intelligence and jobs: Evidence from online vacancies. Journal of Labor Economics, 40(S1), S293-S340.
- Babina, T., Fedyk, A., He, A. X., & Hodson, J. (2020). Artificial Intelligence, Firm Growth, and Industry Concentration. SSRN Scholarly Paper ID 3651052. Social Science Research Network, Rochester, NY.
- Calvino, F. and L. Fontanelli (2023), “A portrait of AI adopters across countries: Firm characteristics, assets’ complementarities and productivity”, OECD Science, Technology and Industry Working Papers, No. 2023/02, OECD Publishing, Paris, https://doi.org/10.1787/0fb79bb9-en.
- Felten, E. W., Raj, M., & Seamans, R. (2018, May). A method to link advances in artificial intelligence to occupational abilities. In AEA Papers and Proceedings (Vol. 108, pp. 54-57).
- Felten, E. W., Raj, M., & Seamans, R. (2023). Occupational heterogeneity in exposure to generative ai. Available at SSRN 4414065. 
- Gmyrek, P., Berg, J., & Bescond, D. (2023). Generative AI and jobs: A global analysis of potential effects on job quantity and quality. ILO Working Paper, 96.
