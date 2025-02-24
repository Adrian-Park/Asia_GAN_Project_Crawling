# -*- coding: utf-8 -*-
"""preprocess.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wJTLmo_1M8U7BMI4u5D9-3UrF67O0LWV
"""

import pandas as pd
from konlpy.tag import Okt
import re

df = pd.read_csv('./crawling/reviews/reviews_2016.csv', index_col = 0)
print(df.head())

# 형태소 분리
okt = Okt()

# 불용어 ]
stopwords = pd.read_csv('./crawling/stopwords.csv', index_col=0)
print(stopwords.head())

movie_stopwords = ['영화', '배우', '감독']
stopwords_list = list(stopwords.stopword) + movie_stopwords # stopwords 컬럼 list형식으로 변환 후 새로운 stopwords list와 결합

# reviews 컬럼에서 문장을 하나씩 추출
count = 0
cleaned_sentences = []
for sentence in df.reviews:
    count += 1
    if count % 10 == 0: # 문장 10개 처리시 .
        print('.', end='')
    if count % 100 == 0: # 문장 100개 처리시 줄바꿈
        print('')
    sentence = re.sub('[^가-힣 | ' ']', '', sentence) # 리뷰 하나 꺼내 sentence로 받음
    token = okt.pos(sentence, stem=True) # 형용사 동사는 원형으로
    df_token = pd.DataFrame(token, columns=['word', 'class']) # 형태소와 품사 분류
    df_cleaned_token = df_token[(df_token['class'] == 'Noun') | # 조건 인덱싱
                        (df_token['class'] == 'Verb') |
                        (df_token['class'] == 'Adjective')]
    words = []
    for word in df_cleaned_token['word']: # 불용어 제거
        if len(word) > 1:
            if word not in stopwords_list:
                words.append(word)
    cleaned_sentence = ' '.join(words)
    cleaned_sentences.append(cleaned_sentence)
df['cleaned_sentences'] = cleaned_sentences # 새 컬럼 생성
print(df.head())

print(df.info())

df = df[['titles', 'cleaned_sentences']] # 타이틀과 전처리가 된 문장만 뽑아내기
print(df.info())
df.to_csv('./crawling/cleaned_review/cleaned_review_2017.csv') # 재저장

