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

        with open(f'./m_data/images/meat/{n}.jpg', 'rb') as file:
            contents = file.read()
            data_url = base64.b64encode(contents).decode('utf-8')

          # 요리 제목과 링크 표시
        response_text += '{}.{}<br> {}<br>'.format(i+1, text.strip(), f'<div><a href=https://www.10000recipe.com/recipe/{n}><img alt="image" src="data:image/jpg;base64,{data_url}" style="width: 20%; height: auto;"></a></div>')

    return response_text

# vectorDB 가져오기.
embedding = OpenAIEmbeddings()
db = Chroma( persist_directory='./DB/meat',  embedding_function=embedding)

retriever = db.as_retriever(search_kwargs={'k':5})

qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0),
    chain_type = 'stuff',
    retriever = retriever,
    return_source_documents = True)


def main():

  #채팅 공간
  if "messages_meat" not in st.session_state:
      st.session_state['messages_meat'] = []

  #초기화 버튼
  if 'check_reset_meat' not in st.session_state:
     st.session_state['check_reset_meat'] = False
    
  # Set page title
  st.set_page_config(
      page_title="육류",
      layout="wide",
      page_icon= ":cut_of_meat:",
  )

  st.title("Ready for Meat Cooding!")
  st.markdown('---')
  st.subheader(''' 주 메뉴인 육류로 요리를 해봐요! 레디포 쿠딩이 빠르게 도와줄거예요  
TIP! 육류를 빼고 입력해도 육류와 어울리는 레시피를 추천해 줄거에요 😊 
''')

  with st.sidebar:
      # add_logo()
    img = Image.open('./img_streamlit/base.jpeg')
    st.image(img)
    st.sidebar.markdown('---')

  #초기화 버튼 눌럿을때
  if st.sidebar.button(label = '초기화'):
      st.session_state['messages_meat'] = []
      st.session_state['check_reset_meat'] = True
  # st.sidebar.markdown('---')
      
# -----------------------------------------------------------#
  # Display chat messages from history on app rerun
  for message in st.session_state['messages_meat']:
      with st.chat_message(message["role"]):
          st.markdown(message["content"], unsafe_allow_html=True)

  # React to user input
  if user_input := st.chat_input("말씀해주세요."):
      # Display user message in chat message container
      with st.chat_message("user"):
          st.markdown(user_input)
          
      # Add user message to chat history
      st.session_state['messages_meat'].append({"role": "user", "content": user_input})

      # Request chat completion
      chatbot_response = qa_chain(user_input)
      response1 = get_chatbot_response1(chatbot_response)

      # Display assistant response in chat message container
      with st.chat_message("assistant"):
          st.markdown(response1, unsafe_allow_html=True)

      # Add assistant response to chat history
      st.session_state['messages_meat'].append({"role": "assistant", "content": response1})
if __name__=='__main__':
  main()