# 한국형 숙련사전 구축 및 구인공고 매칭 프로젝트

본 프로젝트는 텍스트 분석을 통해 한국형 숙련사전을 구축하고, 이를 기반으로 온라인 구인공고(OJP)와 매칭하는 시스템을 개발하는 것을 목표로 합니다.

## 프로젝트 개요

- 한국노동연구원(KLI)의 온라인 구인공고 데이터를 활용
- 텍스트 분석을 통한 한국형 숙련사전 구축
- 구축된 숙련사전을 활용한 구인공고 자동 매칭 시스템 개발

## 주요 기능

1. 숙련사전 구축
   - 구인공고 데이터에서 숙련 관련 키워드 추출
   - LLM을 활용한 숙련 분류 및 정의 생성
   - DBSCAN 클러스터링을 통한 유사 숙련 통합

2. 구인공고 매칭
   - 정규표현식 기반의 키워드 매칭
   - 유사도 계산을 통한 관련성 높은 숙련 선별
   - 대규모 구인공고 데이터 처리 최적화

## 사용된 기술

- Python
- pandas, numpy: 데이터 처리
- NLTK, spaCy: 자연어 처리
- Sentence-BERT: 텍스트 임베딩
- scikit-learn: DBSCAN 클러스터링
- OpenAI API, Anthropic API: LLM 활용
- re, rapidfuzz: 정규표현식 및 문자열 매칭

## 설치 및 사용 방법

1. 필요한 라이브러리 설치:
   ```
   pip install -r requirements.txt
   ```

2. 환경 변수 설정:
   - OpenAI API 키 및 Anthropic API 키 설정

3. 숙련사전 구축:
   ```
   python build_skill_dictionary.py
   ```

4. 구인공고 매칭:
   ```
   python match_job_postings.py
   ```

## 프로젝트 구조

```
project/
│
├── data/
│   ├── raw/              # 원본 구인공고 데이터
│   └── processed/        # 처리된 데이터
│
├── src/
│   ├── preprocessing/    # 데이터 전처리 스크립트
│   ├── skill_extraction/ # 숙련 추출 관련 스크립트
│   ├── clustering/       # 클러스터링 관련 스크립트
│   └── matching/         # 구인공고 매칭 스크립트
│
├── notebooks/            # 분석 및 실험용 Jupyter 노트북
│
├── results/              # 결과 파일
│
├── requirements.txt      # 필요한 라이브러리 목록
│
└── README.md             # 프로젝트 설명 문서
```

## 기여 방법

프로젝트에 기여하고 싶으시다면, 다음 절차를 따라주세요:

1. 이 저장소를 포크합니다.
2. 새 브랜치를 생성합니다 (`git checkout -b feature/AmazingFeature`).
3. 변경사항을 커밋합니다 (`git commit -m 'Add some AmazingFeature'`).
4. 브랜치에 푸시합니다 (`git push origin feature/AmazingFeature`).
5. Pull Request를 생성합니다.

## 라이선스

이 프로젝트는 [MIT 라이선스](https://choosealicense.com/licenses/mit/)하에 배포됩니다.

## 연락처

프로젝트 책임자: [이름] - email@example.com

프로젝트 링크: [https://github.com/yourusername/korean-skill-dictionary](https://github.com/yourusername/korean-skill-dictionary)
