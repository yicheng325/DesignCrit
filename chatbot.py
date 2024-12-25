import streamlit as st
import os
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

st.title("選禮達人")
st.write("每次送禮都苦惱該選什麼才最合對方心意？交給「選禮達人」就對了！只要輸入對象的性別、喜好、預算及場合，我們的 AI 便能從海量商品中迅速幫你精選出最貼合需求的優質禮物，還能提供一站式包裝與配送服務。用最省時、省力的方式，傳遞你最真摯的心意～")

conversational_memory_length = 10
memory = ConversationBufferWindowMemory(k = conversational_memory_length)


if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
else:
    for message in st.session_state.chat_history:
        memory.save_context({'input':message['human']}, {'output':message['AI']})

system_prompt = "你是一個友善且有幫助的助手，請用中文回答用戶的問題。"

groq_chat = ChatGroq(
    groq_api_key = os.environ.get("GROQ_API_KEY"),
    model_name = "llama3-70b-8192"
)

conversation = ConversationChain(
    llm = groq_chat,
    memory = memory,
)

user_question = st.text_input("問我問題:")

if user_question:
    response = conversation(user_question)
    message = {'system':system_prompt, 'human':user_question, 'AI':response['response']}
    st.session_state.chat_history.append(message)
    st.write("選禮達人:", response['response'])