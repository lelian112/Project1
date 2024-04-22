from streamlit_extras.app_logo import add_logo
import streamlit.components.v1 as components
import streamlit as st
from PIL import Image, ImageEnhance
import re
# import openai
import streamlit as st
import os, tenacity
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
# import ast
import streamlit.components.v1 as components
# from openai.embeddings_utils import get_embedding
from streamlit_chat import message
# import chromadb
# import langchain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains import ChatVectorDBChain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from collections import Counter
from scipy import spatial
from langchain_openai import ChatOpenAI
import base64

os.environ["OPENAI_API_KEY"]='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'

### 요약 + 요리제목 5가지
def get_chatbot_response1(chatbot_response):
    response_text = f'{chatbot_response["result"].strip()}<br>추천하는 요리:<br>'

    for i, source in enumerate(chatbot_response['source_documents']):
        s = source.metadata['source']
        numbers = re.findall(r'\d+', s)
        n = ''.join(numbers)

        with open(f"./{source.metadata['source']}", "r", encoding="utf-8") as f:
            text = f.readlines()[0]

        with open(f'./m_data/images/etc/{n}.jpg', 'rb') as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode('utf-8')

          # 요리 제목과 링크 표시
        response_text += '{}.{}<br> {}<br>'.format(i+1, text.strip(), f'<div><a href=https://www.10000recipe.com/recipe/{n}><img alt="image" src="data:image/jpg;base64,{data_url}" style="width: 20%; height: auto;"></a></div>')

    return response_text

# vectorDB 가져오기.
embedding = OpenAIEmbeddings()
db = Chroma( persist_directory='./DB/etc',  embedding_function=embedding)

retriever = db.as_retriever(search_kwargs={'k':5})

qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0),
    chain_type = 'stuff',
    retriever = retriever,
    return_source_documents = True)


def main():

  #채팅공간
  if "messages_etc" not in st.session_state:
      st.session_state['messages_etc'] = []

  #초기화 버튼
  if 'check_reset_etc' not in st.session_state:
     st.session_state['check_reset_etc'] = False
    
  # Set page title
  st.set_page_config(
      page_title="기타",
      layout="wide",
      page_icon= ":chopsticks:",
  )

  st.title("Ready for Cooding!")
  st.markdown('---')
  st.subheader(''' 카테고리에 없는 메인재료를 입력해 보세요! 레디포 쿠딩이 빠르게 도와줄거예요
''')

  with st.sidebar:
      # add_logo()
    img = Image.open('./img_streamlit/base.jpeg')
    st.image(img)
    st.sidebar.markdown('---')

  #초기화 버튼 눌럿을때
  if st.sidebar.button(label = '초기화'):
      st.session_state['messages_etc'] = []
      st.session_state['check_reset_etc'] = True
  # st.sidebar.markdown('---')
      
# -----------------------------------------------------------#
  # Display chat messages from history on app rerun
  for message in st.session_state['messages_etc']:
      with st.chat_message(message["role"]):
          st.markdown(message["content"], unsafe_allow_html=True)

  # React to user input
  if user_input := st.chat_input("말씀해주세요."):
      # Display user message in chat message container
      with st.chat_message("user"):
          st.markdown(user_input)

      # Add user message to chat history
      st.session_state['messages_etc'].append({"role": "user", "content": user_input})

      # Request chat completion
      chatbot_response = qa_chain(user_input)
      response1 = get_chatbot_response1(chatbot_response)

      # Display assistant response in chat message container
      with st.chat_message("assistant"):
          st.markdown(response1, unsafe_allow_html=True)

      # Add assistant response to chat history
      st.session_state['messages_etc'].append({"role": "assistant", "content": response1})

if __name__=='__main__':
  main()