# 만개의 레시피 크롤링

# ===== import ===== #

# selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException
# 로그
from log import Log

# 그외 기본 모듈
import csv
import os 
import time

log = Log('만개의 레시피').logger

log.info("크롤링 시작")

# ===== 셀레니움 설정 ===== #
headless = True
header = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'

# chrome driver option 생성
driver_option = webdriver.ChromeOptions()

# Selenium headless 설정
if headless:
  driver_option.headless = True
elif not headless:
  driver_option.headless = False

# Selenium Option 설정
options = {
  'disable_gpu' : 'disable-gpu',
  'lang'        : 'lang=ko_KR',
  'User_Agent'  : f'user-agent={header}',
  'window-size' : '1920x1080'
}
driver_option.add_argument("--headless")
for i in options.values() :
  driver_option.add_argument(i)

# 불필요한 에러 메세지 삭제
driver_option.add_experimental_option("excludeSwitches", ["enable-logging"])
driver_option.add_experimental_option('excludeSwitches', ['enable-automation'])
driver_option.add_experimental_option('useAutomationExtension', False)
# 브라우저 꺼짐 방지 옵션
driver_option.add_experimental_option("detach", True)

# 크롬 드라이버 버전 설정: 버전명시 안하면 최신
service = ChromeService(ChromeDriverManager().install())

# 크롬 드라이버 실행
driver = webdriver.Chrome(service=service, options=driver_option) 

# ===== 실제 로직 구현 ===== #



def get_recipe_url():
  url_list = []
  try:
    url = 'https://www.10000recipe.com/recipe/list.html?cat3=47&order=reco&page=1'

    # 레시피 목록 페이지 ul 태그
    content_ul_tag = 'common_sp_list_ul'

    # 각 레시피 리스트 태그
    content_li_tag = 'common_sp_list_li'

    # 레시피 리스트 가져오기

    url_list = []

    for idx in range(1, 11) :
      log.info(f'{idx}번째 시작')
      driver.get(f'{url}{idx}')
      recipe_list = driver.find_element(By.CLASS_NAME, content_ul_tag).find_elements(By.CLASS_NAME, content_li_tag)
      log.info(f'{len(recipe_list)}개 가져오기 성공')
      
      for j in range(len(recipe_list)) :
        a_tag = recipe_list[j].find_element(By.CLASS_NAME, 'common_sp_link')
        recipe_url = a_tag.get_attribute('href')
        url_list.append(recipe_url)
      
    
    file_path = 'C:/Users/EZEN/Desktop/projekt1/food_projekt'
    if not os.path.isdir(file_path): 
      os.mkdir(file_path)

    # 파일명 설정
    file_name = '만개의레시피url.csv'

    # 필드 설정
    field_names = ['no', 'url']

    with open(file_path + '/' + file_name, 'w', encoding='utf-8-sig', newline='') as f:
      writer = csv.DictWriter(f, fieldnames=field_names)
      writer.writeheader()

      for idx, j in enumerate(url_list):
        log.info(f'{idx} : {j}')
        writer.writerow({
          'no': idx+1,
          'url': j
        })

    return url_list

  except Exception as e:
    print(e)
    log.error("크롤링 실패")
    
  # finally:
  #   log.info("크롤링 종료")
  #   driver.quit()
  return url_list
# url_list = get_recipe_url()///

def get_recipe_content(url_list):
  file_path = 'C:/Users/EZEN/Desktop/projekt1/food_projekt'
  file_name = 'recipe_contents_rice.csv'
  full_file_path = os.path.join(file_path, file_name)
  field_names = ['제목', '재료', '조리순서']

  with open(full_file_path, 'a', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()


    # 전체 url_list
    for idx, url in enumerate(url_list):
      log.info(f'{idx} 번째 시작')
      # if idx == 3:
      #   break
      driver.get(url)
      try:
        # 제목 가져오기
        title = driver.find_element(By.CLASS_NAME, 'view2_summary').find_element(By.TAG_NAME, 'h3').text
        log.info(f'{title}')
        content_ul = driver.find_element(By.CLASS_NAME, 'ready_ingre3').find_elements(By.TAG_NAME, 'ul')
        all_step = driver.find_element(By.CLASS_NAME, 'view_step').find_elements(By.CLASS_NAME, 'view_step_cont')
      
        # 재료 가져오기 -> 전체
        item_list = []
        for j in content_ul:
          content_li = j.find_elements(By.TAG_NAME, 'li')
          
          sub_title = j.find_element(By.TAG_NAME, 'b').text
          sub_title = sub_title.replace('[', '').replace(']', '')
          log.info(f'sub_title: {sub_title}')

          for k in content_li:
            item = k.find_element(By.TAG_NAME, 'a').text
            unit = k.find_element(By.TAG_NAME, 'span').text
            item = f'{item} {unit}'
            item_list.append(item)
            log.info(f'item: {item}')
          
        result = ", ".join(item_list)
        # 재료 가져오기 끝

        # 조리 순서 가져오기
        step_list = []
        for j in all_step:
          step = j.find_element(By.CLASS_NAME, 'media-body').text
          step = step.split('\n')[0]
          step_list.append(step)
          log.info(f'step: {step}')
        step_result = "\n".join(step_list)
        writer.writerow({
            '제목': title,
            '재료': result,
            '조리순서': step_result
          })
        log.info(f'{idx} 번째 {title} 끝')
        print()
      except NoSuchElementException as e:
        log.error(f'{e}')
        continue
  driver.quit()
  log.info("크롤링 종료")


url_list = get_recipe_url()
get_recipe_content(url_list)


