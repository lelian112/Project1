###### 기본 정보 설정 단계 #######
from fastapi import Request, FastAPI
import openai
import threading
import time
import queue as q
import os
import pandas as pd
import random

# OpenAI API KEY
API_KEY = ''
openai.api_key = API_KEY


###### 기능 구현 단계 #######

# 메세지 전송
def textResponseFormat(bot_response):
    print('textResponseFormat')
    print(bot_response)
    
    response = {'version': '2.0', 'template': {
    'outputs': [{"simpleText": {"text": bot_response}}], 'quickReplies': []}}
    print()
    print(response)
    return response

# # 사진 전송
# def imageResponseFormat(bot_response,prompt):
#     output_text = prompt+"내용에 관한 이미지 입니다"
#     response = {'version': '2.0', 'template': {
#     'outputs': [{"simpleImage": {"imageUrl": bot_response,"altText":output_text}}], 'quickReplies': []}}
#     return response

# 응답 초과시 답변
def timeover():
    response = {"version":"2.0","template":{
      "outputs":[
         {
            "simpleText":{
               "text":"밥순이가 레시피를 생각하고 있어요🙏🙏\n잠시후 아래 말풍선을 눌러주세요👆"
            }
         }
      ],
      "quickReplies":[
         {
            "action":"message",
            "label":"생각 다 끝났나요?🙋",
            "messageText":"생각 다 끝났나요?"
         }]}}
    return response

# ChatGPT에게 질문/답변 받기
def getTextFromGPT(prompt):
    messages_prompt = [
                        {"role": "system", "content": 'You are a thoughtful assistant. Respond to all input in 30 words'},
                        {"role": "assistant", "content": 'Summarize to korean'},
                        {"role": "user", "content": f"{prompt} 를 사용하는 요리법을 찾아줘"}
                      ]
    # messages_prompt = [{"role": "system", "content": 'You are a thoughtful assistant.'}]
    messages_prompt += []
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages_prompt)
    print('response', response)
    message = response["choices"][0]["message"]["content"]
    return message
# # DALLE.2에게 질문/그림 URL 받기
# def getImageURLFromDALLE(prompt):
#     response = openai.Image.create(prompt=prompt,n=1,size="512x512")
#     print('image response:', response)
#     image_url = response['data'][0]['url']
#     return image_url

# 텍스트파일 초기화
def dbReset(filename):
    with open(filename, 'w') as f:
        f.write("")

###### 서버 생성 단계 #######
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "kakaoTest"}

@app.post("/chat/")
async def chat(request: Request):
    kakaorequest = await request.json()
    print('kakaorequest:', kakaorequest)
    return mainChat(kakaorequest)



#######################################################################
# CSV 파일 로드 및 데이터 처리
def load_csv_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("CSV 파일이 성공적으로 로드되었습니다.")
        print(data.head())  # 데이터의 처음 몇 줄을 출력
        return data
    except Exception as e:
        print("CSV 파일 로딩 중 오류 발생:", e)
        return None



# 예시 데이터프레임
data = {
    'text': ['What is AI?', 'How to code in Python?', 'What is data science?'],
    'text': ['AI is the field of study that simulates human intelligence in machines.', 
               'You can learn Python coding from various online resources like tutorials.', 
               'Data science is a multi-disciplinary field that uses scientific methods to extract knowledge from data.']
}

df = pd.read_csv('1000recipe.csv', names=['text'])


# keywords = ['파이썬', '재밌어요']
# result = df['텍스트'].str.contains('|'.join(keywords))

# 질문에 대한 답변 찾기
def find_answer_in_csv(dataframe, query):
    
    # query_words = query.split()  # 쿼리를 단어로 분리합니다.
    
    # 모든 쿼리 단어가 포함된 행을 필터링합니다.
    mask = dataframe['재료'].apply(lambda x: all(word in x for word in query))
    similar_rows = dataframe[mask]
    
    print(similar_rows)
    # similar_rows = dataframe[dataframe['text'].str.contains('&'.join(query), case=False, na=False)]
    
    # 모든 재료가 있을 때
    if not similar_rows.empty:
                            
        range_num = len(similar_rows)         
        random_number = random.randint(0, range_num - 1)
        
        data = similar_rows.iloc[random_number]
        return f"🧑‍🍳입력하신 재료가 전부 있는 요리는 {range_num}개에요\n\n🍰{data['제목']}\n\n[재료]\n{data['재료']}\n\n[조리순서]\n{data['조리순서']}🍝"
    
    # 모든 재료가 없을 때
    else:
         
        similar_rows = dataframe[dataframe['재료'].str.contains('|'.join(query), case=False, na=False)]
        
        
        # 입력한 재료중에 하나만 있을 때
        if not similar_rows.empty:
            range_num = len(similar_rows)
            random_number = random.randint(0, range_num)
            # similar_rows.iloc[random_number]['제목']
            data = similar_rows.iloc[random_number]
            return f"입력하신 재료가 전부 없고 일부만 있는 요리는 {range_num}개에요\n\n[제목]\n{data['제목']}\n\n[재료]\n{data['재료']}\n\n[조리순서]\n{data['조리순서']}"
    
        # 입력한 재료중에 아무것도 없을 때    
        else:  
            bot_res = getTextFromGPT(query)
            return bot_res
        
# CSV 파일 로드
csv_data = load_csv_data("1000recipe.csv")


    

##########################################################################




###### 메인 함수 단계 #######

# 메인 함수
def mainChat(kakaorequest):

    run_flag = False
    start_time = time.time()

    # 응답 결과를 저장하기 위한 텍스트 파일 생성
    cwd = os.getcwd()
    filename = cwd + '/botlog.txt'
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("")
    else:
        print("File Exists")    

    # 답변 생성 함수 실행
    response_queue = q.Queue()
    request_respond = threading.Thread(target=responseOpenAI,
                                        args=(kakaorequest, response_queue,filename))
    request_respond.start()

    # 답변 생성 시간 체크
    while (time.time() - start_time < 3.5):
        if not response_queue.empty():
            # 3.5초 안에 답변이 완성되면 바로 값 리턴
            response = response_queue.get()
            run_flag= True
            break
        # 안정적인 구동을 위한 딜레이 타임 설정
        time.sleep(0.01)

    # 3.5초 내 답변이 생성되지 않을 경우
    if run_flag== False:     
        response = timeover()
    print()
    print(response)
    return response

# 답변/사진 요청 및 응답 확인 함수
def responseOpenAI(request, response_queue, filename):
    if '생각 다 끝났나요?' in request["userRequest"]["utterance"]:
        with open('botlog.txt', 'r') as f:
            last_update = f.read()
            print(last_update)
        # if len(last_update.split()) > 1:
            # bot_res = last_update.split()[1]
            response_queue.put(textResponseFormat(last_update))
            dbReset(filename)
    elif '밥순아' in request["userRequest"]["utterance"]:
        dbReset(filename)
        prompt = request["userRequest"]["utterance"].replace("밥순아", "").strip()
        query = prompt.split(',')
        csv_answer = find_answer_in_csv(csv_data, query)
        if csv_answer:
            
            # bot_res = getTextFromGPT(csv_answer)
            # print(bot_res)
            
            res = textResponseFormat(csv_answer)
            print()
            print(res)
            response_queue.put(res)
            log = "밥순아" + " " + str(csv_answer)
            
        elif csv_answer == None:
            bot_res = getTextFromGPT(prompt)
            response_queue.put(textResponseFormat(bot_res))
            log = "밥순아" + " " + str(bot_res)
            
        save_log = "밥순아" + " " + str(log)
        
        with open(filename, 'w') as f:
            f.write(save_log)
            print('저장완료')
    else:
        base_response = {'version': '2.0', 'template': {'outputs': [], 'quickReplies': []}}
        response_queue.put(base_response)

        
 