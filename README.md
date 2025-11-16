# 📘 Emotion Diary --- Streamlit 감성 일기 앱

당신만의 비밀스러운 감성 일기 ✨\
LLM 기반 AI 비밀친구가 당신의 일기에 따뜻한 대답을 보내주는 Streamlit
앱입니다.

------------------------------------------------------------------------

## 📌 프로젝트 소개

Emotion Diary는 사용자가 작성한 일기 내용을 기반으로\
LLM(ChatGPT API 기반)이 **동물의 숲 너굴** 감성으로 공감과 위로의 답장을
보내주는 감성 일기 서비스입니다.

-   매일의 일기를 작성하면 따스한 AI 답장 자동 생성
-   나만의 일기 리스트 조회
-   클릭 시 해당 일기 상세 페이지 이동
-   CSV 파일로 간단한 로컬 데이터 저장
-   Streamlit UI 기반 간단하고 가벼운 실행

------------------------------------------------------------------------

## ✨ 주요 기능

### 📝 1. 일기 쓰기

-   날짜 입력\
-   일기 내용 입력\
-   저장 시 AI 답장 자동 생성 후 CSV 저장

### 🤖 2. AI 감성 답장

-   LangChain + ChatOpenAI(gpt-4o-mini) 사용\
-   너굴(동물의 숲) 스타일 감성 답장 생성\
-   공감 중심의 따뜻한 말투

### 📂 3. 사이드바 일기 목록

-   날짜 + 미리보기 표시\
-   클릭 시 해당 일기 상세 조회

### 🔍 4. 상세 조회 페이지

-   전체 일기 보기\
-   AI 답장 보기\
-   "일기쓰기" 버튼으로 돌아가기

------------------------------------------------------------------------

## 📁 디렉토리 구조

    emotion-diary/
    │
    ├── diary_entries.csv
    ├── .env
    ├── app.py
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

## 🔧 설치 및 실행 방법

### 1) 레포지토리 클론

``` bash
git clone https://github.com/yourname/emotion-diary.git
cd emotion-diary
```

### 2) 가상환경 설정 (선택)

``` bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate       # Windows
```

### 3) 패키지 설치

``` bash
pip install -r requirements.txt
```

### 4) 환경변수 (.env) 설정

프로젝트 루트에 `.env` 파일 생성:

    OPENAI_API_KEY=your_api_key_here

### 5) 앱 실행

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 🧠 기술 스택

-   **Python 3.9+**
-   **Streamlit**
-   **LangChain**
-   **ChatOpenAI (GPT-4o-mini)**
-   **Pandas**
-   **dotenv**

------------------------------------------------------------------------

## 📦 requirements.txt (예시)

    streamlit
    pandas
    python-dotenv
    langchain
    langchain-openai
    langchain-community

------------------------------------------------------------------------

## 🌟 향후 확장 아이디어

-   로그인 시스템 (Firebase / Supabase)
-   감정 분석 그래프 추가
-   태그 분류 시스템
-   캘린더 UI
-   일기 삭제 & 수정 기능
-   클라우드 DB 연동

------------------------------------------------------------------------

## 💌 문의 & 피드백

더 발전된 기능 또는 UI가 필요하면 언제든 요청하세요!
