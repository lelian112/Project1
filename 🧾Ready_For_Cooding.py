
import streamlit as st
import base64
from PIL import Image, ImageEnhance
import re
import openai
import streamlit as st
import os, tenacity
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import ast
import streamlit.components.v1 as components
# from openai.embeddings_utils import get_embedding
from streamlit_chat import message
import chromadb
import langchain
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
from PIL import Image
import PIL.Image as pil
import requests, json
from bs4 import BeautifulSoup

os.environ["OPENAI_API_KEY"]='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'


def get_chatbot_response(chatbot_response):
    url = f"https://www.10000recipe.com/recipe/list.html?q={chatbot_response}"
    response = requests.get(url)
    if response.status_code == 200:
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
    else :
        print("HTTP response error :", response.status_code)
        return

    menus = soup.find_all('li', {'class':'common_sp_list_li'})
    response_text = ""

    for index, menu in enumerate(menus):
      if index >= 5:
        break
      title = menu.find('div', {'class':'common_sp_caption_tit'}).text
      image = menu.find('div', {'class':'common_sp_thumb'}).find('img')['src']
      href = menu.find('div', {'class':'common_sp_thumb'}).find('a')['href']
      response_text += '{}<br>{}'.format(title, f'<div><a href=https://www.10000recipe.com{href}><img alt="image" src="{image}" style="width: 20%; height: auto;"></a></div>')
    
    return response_text

# def get_chatbot_response2(chatbot_response):
#     response_images = ""

#     for i, source in enumerate(chatbot_response['source_documents']):
#         s = source.metadata['source']
#         numbers = re.findall(r'\d+', s)
#         n = ''.join(numbers)

#         with open(f'./m_data/images/pork/{n}.jpg', 'rb') as file:
#             contents = file.read()
#             data_url = base64.b64encode(contents).decode('utf-8')

#             image1 = f'<div><a href=https://www.10000recipe.com/recipe/{n}><img alt="image" src="data:image/jpg;base64,{data_url}" style="width: 20%; height: auto;"></a></div>'
#         response_images += image1

    #return response_images

            # image1 = Image.open(f'./m_data/images/pork/{n}.jpg')
            # imag1_size = image1.size
            # image1 = image1.resize((int(imag1_size[0]*(0.5)), int(imag1_size[1]*(0.5))))
            # text = f.readlines()[0]
            # response_text += '{}.{} \n{}{}\n'.format(st.text(i+1), st.text(text.strip()), st.write(f'https://www.10000recipe.com/recipe/{n}'), st.image(image1),)

# vectorDB 가져오기.
embedding = OpenAIEmbeddings()
db = Chroma( persist_directory='./DB/all',  embedding_function=embedding)

retriever = db.as_retriever(search_kwargs={'k':5})

qa_chain = RetrievalQA.from_chain_type(
    llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature = 0),
    chain_type = 'stuff',
    retriever = retriever,
    return_source_documents = True)

def main():

  #chat
  if "messages" not in st.session_state:
      st.session_state.messages = []

  #초기화 버튼
  if 'check_reset' not in st.session_state:
     st.session_state['check_reset'] = False

  #페이지 기본 설정
  st.set_page_config(page_title = '레디 포 쿠딩', layout = 'wide', initial_sidebar_state="expanded", page_icon=':receipt:')

  #제목
  st.header("Hello, I'm 'Ready For Cooding Bot'")
  st.markdown('---')

  #기본 설명
  with st.expander('<레디 포 쿠딩>은요 ', expanded=True):
    st.write(
      '''
      사용자가 가지고있는 식재료를 기반으로 맞춤형 레시피를 제공하여 다양하고 맛있는 요리 아이디어를 제시합니다. 
      요리 초보자부터 전문가까지 모든 사용자를 위한 요리에 대한 아이디어와 직관적인 경험을 제시합니다. 사용자는 선택된 재료에 기반하여 레시피를 찾을 뿐만 아니라, 난이도, 소요시간, 유형, 이미지까지 다양하게 검색 할 수 있어요. 
      간편하고 즐거운 콜라보레이션을 통하여 <레디 포 쿠딩 봇>은 요리를 더욱 흥미롭게 만들어주고 실용적이며 즐거운 요리 체험을 선사 할 것 입니다. 

      자, 요리 하실 준비 되셨을까요? Ready for Cooding ? Let’s Go! 
      '''
    )
  
  with st.expander('<레디 포 쿠딩 봇> 사용방법', expanded=False):
    st.write(
    ''' 
    “Ready for Cooding Bot”은 간편하고 맞춤형 레시피 검색을 위해 사용자 친화적인 인터페이스를 제공합니다. 간단한 이용방법을 설명합니다. 

    - 메인페이지에서 재료검색 : 
    메인페이지에서는 가용자가 가지고 있는 재료를 자유롭게 입력여 검색버튼을 클릭하면 <레디 포 쿠딩봇>은 해당재료를 기반으로 한 맞춤형 레시피를 즉시 제공합니다. 
    TIP ! : ‘레디포 쿠딩봇’에게 “사과와 딸기를 이용한 레시피를 알려줘” 와 같이 대화하듯이 말해보세요! 당신을 위해 레디포 쿠딩봇은 열심히 요리해 줄거예요 ! 
    - 주 메뉴에서 재료검색 : 
    사이드바에 있는 주 메뉴페이지를 클릭하면 해당 주 메인재료와 사용자가 검색한 재료와 관련된 레시피를 제공합니다. 예를들어 소고기 탭을 선택하고 마늘, 양파가 들어간 레시피를 알려줘! 라고 하면 소고기가 메인이고 마늘과 양파가 들어간 요리를 <레디 포 쿠딩봇>이 빠르게 알려줄거예요! 
    TIP: ‘채소’ 메뉴에서 비건요리 레시피를 찾아보세요! 한층 더 건강해지는 
    느낌을 받을 수 있어요! 

    이러한 사용방법을 통하여 <레디 포 쿠딩 봇>은 사용자에게 편리하고 맛있는 요리를 즐길 수 있는 다양한 옵션을 제공합니다. 

    '''
)
  with st.sidebar:
      # add_logo()
    img = Image.open('./etc/base.jpeg')
    st.image(img)
    st.sidebar.markdown('---')

  #초기화 버튼 눌럿을때
  if st.sidebar.button(label = '초기화'):
      st.session_state.messages = []
      st.session_state['check_reset'] = True
  # st.sidebar.markdown('---')

  # Display chat messages from history on app rerun
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.markdown(message["content"], unsafe_allow_html=True)

  # React to user input
  if user_input := st.chat_input("요리에 관한 모든것을 물어보세요!"):
    #   avatar_image = './img_streamlit/chatbot.png'
      # Display user message in chat message container
      with st.chat_message("user"):
          st.markdown(user_input)

      # Add user message to chat history
      st.session_state.messages.append({"role": "user", "content": user_input})

      chatbot_response = user_input
      # Request chat completion
      response1 = get_chatbot_response(chatbot_response)

      # Display assistant response in chat message container
      with st.chat_message("assistant"):
          st.markdown(response1, unsafe_allow_html=True)

      # Add assistant response to chat history
      st.session_state.messages.append({"role": "assistant", "content": response1})

    #   role = message["role"]

    #   if role == "assistant":
    #     avatar_image = './img_streamlit/chatbot.png'
    #   elif role == "user":
    #     avatar_image = './img_streamlit/chatbot.png'
    #   else:
    #     avatar_image = None

    #   with st.chat_message(role, avatar=avatar_image):
    #     st.write(message["content"])

if __name__=='__main__':
  main()
