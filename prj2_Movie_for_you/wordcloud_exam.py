# 모듈 임포트!
import collections

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib as mpl
from matplotlib import font_manager, rc

# 폰트와 csv 파일 불러오기
fontpath = './malgun.ttf'
font_name = font_manager.FontProperties(fname=fontpath).get_name()
rc('font', family=font_name)
mpl.font_manager._rebuild()

df = pd.read_csv('./crawling/one_sentence_review_2016_2021.csv', index_col=0)
df.dropna(inplace=True) # nan값 제거
# print(df.info())
#
# print(df.head())

# 지정 인덱스 값의 제목이 같은 로우 찾기
movie_index = df[df['titles'] == '슈퍼 햄찌 (SUPER FURBALL)'].index[0]
print(movie_index)
print(df.reviews[movie_index])

# 단어 리스트에 띄어쓰기 기준으로 단어 잘라내어 담기
words = df.reviews[movie_index].split(' ')
print(words)

# 단어사전 생성
worddict = collections.Counter(words) # 단어 빈도 카운트
worddict = dict(worddict) # dict 형태로 변환
print(worddict)

# 워드클라우드에서 제외할 단어 선정
stopwords = ['관객', '작품', '주인공', '개봉', '촬영', '출연']

# stopword 적용 O
wordcloud_img = WordCloud(background_color='white', max_words=2000,
                          font_path=fontpath,
                          collocations=False,
                          stopwords=stopwords).generate(df.reviews[movie_index])
# stopword 적용 X
wordcloud_img = WordCloud(background_color='white', max_words=2000,
                          font_path=fontpath,).generate_from_frequencies(worddict)

plt.figure(figsize=(8,8))
plt.imshow(wordcloud_img, interpolation='bilinear')
plt.axis('off') # 눈금 X
plt.title(df.reviews[movie_index], size=25)
plt.show()
