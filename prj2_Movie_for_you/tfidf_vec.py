# 모듈 임포트!
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite, mmread # 매트리스 형식으로 저장하는 모듈
import pickle

df_review_one_sentences = pd.read_csv('./crawling/one_sentence_review_2017_2020.csv', index_col=0)
print(df_review_one_sentences.info())

Tfidf = TfidfVectorizer(sublinear_tf=True)
Tfidf_matrix = Tfidf.fit_transform(df_review_one_sentences['reviews'])

# 새로운 영화 리뷰 Tfidf에 추가하기 위해선 transform된 Tfidf를 가지고 있어야 함

# Tfidf 피클로 저장
with open('./models/tfidf.pickle', 'wb') as f:
    pickle.dump(Tfidf, f)

mmwrite('./models/tfidf_movie_review.mtx', Tfidf_matrix)
