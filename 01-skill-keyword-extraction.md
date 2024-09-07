### 단어 추출 절차
#### 구인공고 샘플링
  - 전체 5,000,000 건 중 5% (N = 200,000) 랜덤 샘플링
  - 수집된 업종 정보를 처리하여 구인공고의 업종별 분포 확인
![[Pasted image 20240907211953.png]]
#### 구인공고 전처리

![[Pasted image 20240907212019.png]]
  - 숙련과 무관한 내용을 미리 제거함.
  - 토큰 수 감소와 추출 품질 개선을 위해 필수적. 잘못된 추출을 방지하므로 추출 뒤 전처리의 수고를 크게 덜 수 있음.
  - 구인공고는 '직무 관련' 내용이 나오고 '근무 조건', '복리 후생', '전형 절차' 내용이 나오는 구조화된 텍스트이므로 이러한 전처리가 가능함.

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
        r"유의\s*사항\s*[ㆍ·]\s*학력[,\s]*성별[,\s]*연령을\s*보지\s*않는\s*블라인드\s*채용입니다\.?",
        r"입사\s*지원\s*서류에\s*허위\s*사실이\s*발견될\s*경우[,\s]*채용\s*확정\s*이후라도\s*채용이\s*취소될\s*수\s*있습니다\.?",
        r"모집\s*분야별로\s*마감일이\s*상이할\s*수\s*있으니\s*유의하시길\s*바랍니다\.?",
        r"상세\s*사항\s*전화\s*문의\s*요망",
        r"접수된\s*지원서는\s*최초\s*접수일로부터\s*1년간\s*보관되며\s*1년이\s*경과된\s*뒤에는\s*자동\s*파기",
        r"급여\s*및\s*복리후생은\s*당사\s*규정에\s*준하여\s*적용",
        r"3개월\s*근무\s*평가\s*후\s*정규직\s*전환\s*면접\s*실시",
        r"이력서\s*등\s*제출\s*서류에\s*허위\s*사실이\s*있을\s*시\s*채용\s*취소",
        r"자세한\s*상세\s*요강은\s*반드시\s*채용\s*홈페이지에서\s*직접\s*확인해\s*주시기\s*바랍니다"
    ]
    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    # 절사 기준이 되는 문구 인식
    base_keywords = [
        r'혜택\s*및\s*복지',
        r'근무\s*조건',
        r'전형\s*절차',
        r'접수\s*기간',
        r'지원\s*방법',
        r'지원\s*자격',
        r'채용\s*절차',
        r'제출\s*서류',
        r'기타\s*사항',
        r'복지\s*제도',
        r'급여\s*및',
        r'근무\s*형태',
        r'근무\s*환경',
        r'복리\s*후생',
        r'서류\s*전형',
        r'접수된\s*지원서',
    ]
    cut_keywords = '|'.join(base_keywords)
    pattern = fr'(^|\s|\S)({cut_keywords})'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        text = text[:match.start() + 1] # +1 to include the character before the keyword
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text
```

#### 프롬프트 엔지니어링
  - 추출 방법 : OpenAI API를 활용해 온라인 공고에서 숙련 용어를 추출.
    - GPT 3.5 Instruct
    - GPT 4-o mini
  - 결과물 형식 : 숙련 용어를 구분자 파이프(|)로 구분하여 추출하도록 지시
  - 파라미터 조정 : 응답의 creativity 파라미터인 temperature를 낮추고, 임의 응답을 줄이기 위해 max token 수도 적절하게 제한

- 추출 안정성 개선
  - 작은 크기로 나누어 추출
  - 중간 결과물을 저장하게 하고, 시중단 지점에서 쉽게 재개할 수 있게 함.
  - 안정적인 로컬 환경에서 복수 개의 세션으로 진행