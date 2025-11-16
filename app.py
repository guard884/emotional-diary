import streamlit as st
import pandas as pd
import os

from dotenv import load_dotenv
from datetime import datetime
from langchain_community.chat_models import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationSummaryMemory
from langchain_core.prompts  import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
##############################################################
# LangChain객체생성 함수
def langchain_init():
    # 환경 변수 로드 (.env 파일에서 OPENAI_API_KEY 불러옴)
    load_dotenv(".env")
    # LLM 초기화
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # LCEL 기반 프롬프트
    main_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", """너는 내가쓴일기에 나만을위한 작고소중한 비밀친구가 내감정상태를 살피고 공감과 위로의 답장을 건네는 AI
                    딱딱한 조언을해주는 전문가가 아니라 비밀스러운 펜팔친구처럼 친근한관계를맺는 답장를쓰는 AI
                    친구나가족에게 공유하긴싫고 나혼자보고싶어 근데 또누군가 봐줬으면좋겠는데하는 일기에 답장을하는 AI야.
                    동물의숲 너굴느낌으로 답변해주는 AI야.   
                    """),
        #("user", "오늘 면접 1차 합격이야 기쁘지만 2차면접이 걱정이야, 대학교때는 좋았는데 졸업하니 힘드네요 ㅠㅠㅠ"),
        ("assistant", """동물의 숲 주민 느낌으로 답장해줘!
                        오늘 하루 참 반짝거렸겠다, 너굴~
                        네 이야기 들으니까 나도 괜히 옛날 대학 시절이 살짝 떠오르네.
                        오랜만에 만난 친구랑 맛있는 것도 먹고, 이런저런 얘기도 하고… 그런 시간이 마음을 가볍게 해주지잉~
                        그리고 면접 1차 합격이라니, 정말 축하해! 🎉
                        오늘은 너 스스로도 기분 좋아하는 하루였을 것 같아.
                        이럴 땐 그냥 설렘 가득 안고 푹 쉬면 되는 거야, 알지?
                        공부는… 음, 이런 날에는 너무 욕심 내지 말고, 집에 가서 살짝만 해도 충분해~ 후훗.
                        다음 소식도 기다릴게!
                        잘 자구~ 오늘 이야기를 읽으니까 나도 괜히 마음이 말랑해진다, 너굴너굴 😄
                    """),
        ("user", " {input}"),
        ]
    
    )

    # 3️. 출력 파서 (응답 텍스트만 추출)
    parser = StrOutputParser()
    # 4️. LCEL 파이프라인 구성 (Prompt → LLM → Parser)
    conversation_chain = main_prompt | llm | parser
    return conversation_chain
##############################################################

##############################################################
# 저장함수

# def save_entry(date, entry,answer ):
#     new_entry = pd.DataFrame([{'Date': date, 'Entry': entry,'Answer':answer}])
#     # Append to the CSV file
#     new_entry.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))
def save_entry(date, entry, answer):
    # 기존 데이터 불러오기 (ID 포함)
    df = load_entries()

    # 새 ID 계산: 마지막 ID + 1
    new_id = 1 if df.empty else df['ID'].max() + 1

    new_entry = pd.DataFrame([{
        'ID': new_id,
        'Date': date,
        'Entry': entry,
        'Answer': answer
    }])

    # CSV에 추가 저장
    new_entry.to_csv(CSV_FILE, mode='a', index=False, header=not os.path.exists(CSV_FILE))

##############################################################

##############################################################
# 데이터 로드 함수 (ID 포함)
# def load_entries():
#     if os.path.exists(CSV_FILE):
#         df = pd.read_csv(CSV_FILE)
#         # ID가 없으면 새로 생성하여 추가
#         if 'ID' not in df.columns:
#              df['ID'] = range(1, len(df) + 1)
#              df.to_csv(CSV_FILE, index=False) # ID를 파일에도 저장
#         return df
#     return pd.DataFrame(columns=['Date', 'Entry', 'Answer','ID'])
def load_entries():
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)

        # ID가 없을 때만 생성
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
            df.to_csv(CSV_FILE, index=False)

        return df

    # 파일 없을 경우 빈 df 반환
    return pd.DataFrame(columns=['ID', 'Date', 'Entry', 'Answer'])
##############################################################

##############################################################
# 링크 클릭 시 실행될 콜백 함수
def select_entry(entry_id):
    """클릭된 일기의 ID를 세션 상태에 저장합니다."""
    st.session_state['selected_entry_id'] = entry_id
##############################################################    
# ------------------ Main App Layout ----------------------
conversation_chain=langchain_init()
CSV_FILE = 'diary_entries.csv'

# 세션 상태 초기화
if 'selected_entry_id' not in st.session_state:
    st.session_state['selected_entry_id'] = None


# --------------- 앱 메인레이아웃 시작 ---------------------
st.set_page_config(page_title="Streamlit Diary", layout="centered")
st.title("📘 감성 일기")
st.divider()
# Input widgets
# -------------------------
# 입력 섹션
# -------------------------
if st.session_state['selected_entry_id'] is None:
    st.header("📝 일기쓰기")
    new_date = st.date_input("날짜")
    new_diary = st.text_area("내용", height=200)
    if st.button("저장"):
        if new_diary:
            result = conversation_chain.invoke(new_diary)
            save_entry(new_date, new_diary,result)
            st.success("일기저장 성공!")
            
        else:
            st.warning("일기를 입력하세요")


# --------------- 메인 화면: 선택된 일기 내용 표시 ---------------
df = load_entries()



if st.session_state['selected_entry_id'] is not None:
    selected_id = st.session_state['selected_entry_id']
    # DataFrame에서 해당 ID를 가진 행 찾기 (ID가 숫자인지 확인 필요)
    selected_entry = df[df['ID'] == selected_id]

    if not selected_entry.empty:
        # 시리즈(Series) 형태로 데이터 추출
        entry_data = selected_entry.iloc[0]
        # # 🔹 뒤로가기 버튼
       

        st.subheader(f"날짜:{entry_data['Date']}")
        st.markdown("---")
        # st.markdown을 사용하여 내용을 표시하거나 st.text_area에 넣을 수 있습니다.
        st.markdown(f"일기내용: \n\n{entry_data['Entry']}")
        st.markdown(f"AI답변: \n\n{entry_data['Answer']}")
        if st.button("⬅ 일기쓰기"):
            st.session_state['selected_entry_id'] = None
            st.rerun()
    else:
        st.warning("선택된 일기 데이터를 찾을 수 없습니다.")

else:
    st.subheader("일기 내용을 보려면 왼쪽 목록에서 선택하세요.")



# --------------- 사이드바: 일기 목록 (링크 스타일) ---------------------
st.sidebar.subheader("일기목록")
if not df.empty:
    # 각 행에 대해 링크 버튼 생성
    for index, row in df.iterrows():
        # st.sidebar.link_button 대신 st.sidebar.button을 사용하여 콜백 연결
        # 앵커 태그처럼 보이지는 않지만 클릭 이벤트는 정확히 처리됩니다.
        button_label = f"{row['Date']} - {row['Entry'][:30]}..."

        # 각 버튼에 고유한 키를 부여하고 클릭 시 select_entry 함수 호출
        if st.sidebar.button(
            button_label,
            key=f"link_btn_{row['ID']}",
            on_click=select_entry,
            args=(row['ID'],) # 콜백 함수에 인자로 ID 전달
        ):
            pass # on_click이 실행되므로 여기서는 추가 작업 불필요

else:
    st.sidebar.info("저장된 일기가 없습니다.")

