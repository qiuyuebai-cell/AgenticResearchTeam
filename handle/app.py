import os
import streamlit as st
from crew_setup import create_crew_ai
from ingest_data import ingest_knowledge # 导入我们的数据处理函数

# --- 部署时自动创建数据库 ---
db_path = './trisolaris_db'
if not os.path.exists(db_path):
    st.toast("首次启动：正在初始化知识库...")
    with st.spinner("请稍候，正在为您准备AI的核心记忆..."):
        ingest_knowledge()
    st.toast("知识库准备就绪！")


st.set_page_config(page_title='Agentic Research Team', page_icon='🤖')
st.title('Agentic Research Team')
st.markdown('一个AI驱动的RAG知识系统')

# 初始化会话状态，用于存储聊天记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 获取用户输入
user_prompt = st.chat_input("请输入你关于《三体》的问题...")

if user_prompt:
    # 显示用户消息
    st.chat_message("user").markdown(user_prompt)
    # 添加到历史记录
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # 创建并运行AI Crew
    with st.spinner("AI团队正在协作中，请稍候..."):
        try:
            ai_crew = create_crew_ai(user_prompt)
            result = ai_crew.kickoff()

            # 显示AI回答
            with st.chat_message("assistant"):
                st.markdown(result)
            # 添加到历史记录
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"执行过程中出现错误: {e}")
