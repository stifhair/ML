#데이터를 시각해주는 가장 기본적인 라이브러리
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import requests

import json

from bs4 import BeautifulSoup

import re

restaurant_id_list = []

#100 개씩의 API 호출 결과를 3번, 마지막에 50번을 한번 더, 총 4번 가져옴, 크롤링 260회가 넘어가면 오류가 발생
for start_idx in [1,101,201,251]:
    # 네이버 플레이스 API를 호출하기 위한 기본 주소.
    source_url = "https://stroe.naver.com/sogum/api/businesses?"

    # 검색 규칙 파라미터를 추가.
    url_parameter_start = "start=" + str(start_idx)
    url_parameter_display = "&display=" # start 와 display는 검색결과를 얼마큼 보여줄지에 관련된 파라미터
    url_parameter_query = "&query=홍대+고기집" # 검색하고 싶은 장소나 음식점에 대한 검색어
    url_parameter_sorting = "&sortingOrder=precision" # 어떤 방식으로 검색 결과를 정할지에 대한 파라미터
    url_concat = soruce_url+url_parameter_start + \
                url_parameter_display + str(start_idx + 199) + url_parameter_query + url_parameter_sorting
    print("request_url:", url_concat)

    # 반환받은 API 데이터에 json, loads 함수를 사용
    Json_data = request.get(url_concat).text # request.get 함수는 json 형태의 검색 결과 데이터를 얻을 수 있음
    restaurant_list_data = Json.loads(Json_data) # json.loads 함수는 파이썬의 딕션너리 형태로 사용 가능

    #크롤링에 필요한 각 리뷰 상세 페이지의 id를 추출
    for restaurant in restaurant_list_data['items']:
        if 'moreBookingReviewsPath' in restaurant:
            restaurant_id_list.append(restaurant['id'])

restaurant_id_list = list(set(restaurant_id_list))

columns = ['score', 'review']
df = pd.DataFrame(columns=columns)

# 네이버 리뷰 상세 페이지의 기본 주소
source_url_head = "https://store.naver.com/restaurants/detail?id="
source_url_tail = "&tab=bookingReview#_tab"

for idx in range(0, len(restaurant_id_list)):
    print("doing crawling", str(int(idx/len(restaurant_id_list)*100)), "% complete..")

    # 앞서 추출한 리뷰 상세 페이지의 id를 기본 주소의 파라미터 추가
    req = requests.get(source_url_head + str(restaurant_id_list[idx])+ source_url_tail)
    html = req.content
    soup = BeautifulSoup(html,'lxml')
    review_area = soup.find(name='div', attrs={"class": "review_area"})

    # 리뷰가 없는 페이지는 아무 작업도 수행하지 않음
    if review_area is None:
        continue

    # html 구조에서 리뷰의 함수 , 텍스트 부분을 추출/ 네이버 지식인 참고
    review_list = review_area.find_all(name"div", attrs={"class":"info_area"})
    for review in review_list:
        score=review.find(name="scan",attrs={"class":"score"}).text
        review_txt = review.find(name="div", attrs={"class":"review_txt"}).text

        # 추출한 리뷰의 점수, 리뷰 텍스트를 데이터프레임 병합
        row = [score, review_txt]
        series = pd.Series(row, index= df.columns)
        df = df.append(series,ignore_index=True)
print("crawling finished")

#4점 이상의 리뷰는 긍정-리뷰 , 4점 이하의 리뷰는 부정 리뷰로 평가
#수집이 완료된 데이터 셋을 요양값 살펴보기 , head 함수
df['y'] = df['score'].apply(lambda x:1 if float(x) > 4 else 0)
print(df.shape)
df.head()

df.to_csv("review_data.csv", index =False)
df = pd.read_csv("review_data.csv")

# 한글로 전처리하기 , 분류의 피치로 사용할 수 있도록 텍스트 전처리
# 텍스트 정제함수 : 함글 이외의 문자는 전부 제거
def text_cleaning(text):
    #한글의 정규표현식으로 한글만 추출
    hangul = re.compile('[^ㄱ+')
    result = hangul.sub(',text')
    return result

df['ko_text'] = df['review'].apply(lambda x: test_cleaning(x))
del df['review']
df.head()

#형태소 단위로 추출하는 전처리 과정도 진행
#https://ellun tistory.com/46에서 참고
#텍스트 데이터를 분류 모델에 학습이 가능한 데이터셋으로 만드는 과정

from konlpy.tag import okt

def get_pos(x):
    tagger =okt()
    pos = tagger.pos(x)
    pos = ['{}/{}'.format(word,tag) for word, tag in pos]
    return pos

# 형태소 추출 동작을 테스트
result = get_pos(df['ko_text'][0])
print(result)

from sklearn.feature_extraction.text import CountVectorizer

# 형태소를 벡터 형태의 학습 데이터셋으로 변환
index_vectorizer = CountVectorizer(tokenizer = lambda x: get_pos(x))
x = index_vectorizer.fit_transform(df['ko_text'].tolist())
print(x.shape)

#분류 모델의 학습 데이터로 전환
from sklearn.feature_extraction.text import TfidfTransformer

#TF-IDF 방법으로 , 형태소를 벡터 형태의 학습 데이터셋(X 데이터)으로 변환,
tfldf_vectorizer= TfIdfTransformer()
X =tfidf_vectorizer.fit_transform(x)

#긍정/부정 리뷰 분류 모델링 데이터셋 분리 
# sklearn.model_selection에서 제공하는 train_test_split()함수 사용하여 
# 학습 데이터셋, 테스트용 데이터셋으로 데이터를 분리

from sklearn.model_selection import train_test_split

y=df['y']
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.30)
print(x_train.shape)
print(x_test.shape)

#분류 모델링 : fhwltmxlr ghlrnl ahepf

from sklearn.linear_model import LogisticRegression
from sklearn.netrics import accuracy_score, precision_score, recall_score,f1_score

# 로지스틱 회귀모델을 학습
lr = LogisticRegression(random_state=0)
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
x_pred_probability=-lr.predict_proba(x_test[:,1])

print("accuracy(정확도)")
