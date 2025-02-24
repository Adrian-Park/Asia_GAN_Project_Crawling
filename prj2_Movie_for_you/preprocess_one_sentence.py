import pandas as pd

df = pd.read_csv('./crawling/cleaned_review/cleaned_review_2016.csv', index_col=0)
df.dropna(inplace=True)
df.to_csv('./crawling/cleaned_review_2020.csv')

one_sentences = [] # 빈 리스트 생성
for title in df['titles'].unique(): # 영화 제목들 한 번씩만
    temp = df[df['titles'] == title]['cleaned_sentences']
    one_sentence = ' '.join(temp) # 리뷰들 하나로 이어붙이기
    one_sentences.append(one_sentence)

df_one_sentences = pd.DataFrame({'titles':df['titles'].unique(), 'reviews':one_sentences})
print(df_one_sentences.head())

df_one_sentences.to_csv('./crawling/one_sentece_review/one_sentence_review_2016.csv')

# for idx, title in enumerate(df['titles'].unique()):
#     temp = df[df['titles'] == title]['cleaned_reviews']
#     one_senetece = ' '.join(temp)
#     one_sentence