# 모듈 임포트
import pandas as pd
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmwrite,mmread
import pickle

# 데이터 로드
df_review_one_sentence = pd.read_csv('./crawling/one_sentence_review_2016_2021.csv', index_col=0)

# df.iloc[row, col] row, col 숫자 값으로
# print(df_review_one_sentence.iloc[0]) # row 인덱싱
# print(df_review_one_sentence.iloc[0,0]) # 0번 row 0번 column 확인

# df.loc['tom', 'math'] 컬럼명으로
# print(df_review_one_sentence.loc[:,'reviews']) # 컬럼 인덱싱

# 다른 방법
# print(df_review_one_sentence['titles'][0]) # column 따로 row 따로

# enumerate
# ls = ['겨울왕국', '라이온킹', '알라딘']
# print(list(enumerate(ls)))
# for idx, i in enumerate(ls):
#     if i == '라이온킹':
#         print(idx, i)
# 리스트 = [아이 + ' 짱 재밌당' for 아이 in ls]
# print(리스트)

Tfidf_matrix = mmread('./models/tfidf_movie_review.mtx').tocsr()
with open('./models/tfidf.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

# 함수
def getRecommendation(cosine_sim): # cosine_sim 매개변수, 유사도 비교 (-1 ~ 1 값)
    # 생성된 matrix(문장 유사도) 이용
    simScore = list(enumerate(cosine_sim[-1])) # 데이터 순서대로 각 영화에 인덱스 값을 부여 (후에 영화를 찾기 위해). 맨 마지막인 -1은 아래 Tfidf_matrix
    simScore = sorted(simScore, key=lambda x:x[1], reverse=True) # 코사인값(simScore)이 큰 순으로 정렬(reverse=True)
    simScore = simScore[1:11] # 코사인값이 유사한 10개 선정(0은 자기자신이므로 제외)
    movieidx = [i[0] for i in simScore] # enumerate로 부여된 인덱스값
    recMovieList = df_review_one_sentence.iloc[movieidx] # df_review_one_sentence에서 movieidx 값으로 row 인덱싱
    return  recMovieList

# 영화제목으로 찾기
# movie_idx = df_review_one_sentence[df_review_one_sentence['titles']=='라이온 킹 (The Lion King)'].index[0]

# 영화 인덱스로 검색
movie_idx = 2185
print(df_review_one_sentence.iloc[movie_idx,0])

cosine_sim = linear_kernel(Tfidf_matrix[movie_idx], Tfidf_matrix) # linear_kernel로 코사인 유사도를 구함
recommendation = getRecommendation(cosine_sim) # getRecommendation에 코사인 값을 넘김으로써
# print(recommendation)
# 제목만 보고 싶으면
print(recommendation.iloc[:, 0])