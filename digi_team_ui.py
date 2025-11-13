import streamlit as st
#for uploading files
import pandas as pd
from io import StringIO
#Helps to get the user name
import getpass
from digi_team_llm import chat_with_llm
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.chat_message_histories import SQLChatMessageHistory
uploaded_file =None


st.logo(image=r"images\ds_blue1.png", size="large")
st.title(":blue[Industrial_EngineeringGBT]")
st.write("Hallo, stell mir bitte deine Frage!")

user_id= getpass.getuser()
#user_id= "test_id_1"

def get_session_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if st.button(":blue[Chat löschen]"):
    st.session_state.chat_history = []
    history = get_session_history(user_id)
    history.clear()

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


col1, col2 = st.columns([3, 1])

prompt = st.chat_input("Stell eine Frage")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_excel(uploaded_file)
    st.write(dataframe)

if prompt:
    st.session_state.chat_history.append({'role': 'user', 'content': prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if uploaded_file is not None:
            response = st.write_stream(chat_with_llm(user_id, f"{prompt} Das ist eine Excel Datei mit weiterem Kontext: {dataframe}"))
        else:
            response = st.write_stream(chat_with_llm(user_id, prompt)) 

    st.session_state.chat_history.append({'role': 'assistant', 'content': response})


