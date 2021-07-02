import pandas as pd
from gensim.models import Word2Vec

review_word = pd.read_csv('./crawling/cleaned_review/cleaned_movie_review_2016_2021.csv', index_col=0)
# print(review_word.info())

cleaned_token_review = list(review_word['cleaned_reviews'])
# print(len(cleaned_token_review))

cleaned_tokens = []
count = 0
for sentence in cleaned_token_review:
    token = sentence.split(' ')
    cleaned_tokens.append(token)
# print(len(cleaned_tokens))
# print(cleaned_token_review[0])
# print(cleaned_tokens[0])

# Word2Vec 파라미터 설명 https://hoonzi-text.tistory.com/2
embedding_moel = Word2Vec(cleaned_tokens, vector_size=100, window=4, min_count=20, workers=4, epochs=100, sg=1)
# vector_size 축소할 차원 크기 (단어 하나 당 좌표 100개)
# window 커널사이즈 (훈련 시 앞 뒤로 고려하는 단어 개수)
# min_count 최소 빈도 수. 해당 빈도수보다 적게 등장하면 학습서 배제
# workers CPU 사용 갯수
# epochs 학습 횟수
# sg 어떠한 알고리즘을 사용할 지 (sg는 Skip-gram)

embedding_moel.save('./models/word2VecModel_2016_2021.model')

# print(embedding_model.wv.vocab.keys())
# print(len(embedding_model.wv.vocab.keys()))
