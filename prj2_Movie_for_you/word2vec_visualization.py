# 모듈 임포트!
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl

# 폰트 설정
font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)

embedding_model = Word2Vec.load('./models/word2VecModel_2016_2021.model')
key_word = '고양이'
sim_word = embedding_model.wv.most_similar(key_word, topn=10) # 모델을 이용하여 코사인 값이 가장 유사한 단어 10개 추출
print(sim_word)

# 차원축소
# https://excelsior-cjh.tistory.com/167

# 빈 리스트 생성
vectors = []
labels = []

for label, _ in sim_word:
    labels.append(label)
    vectors.append(embedding_model.wv[label])
df_vectors = pd.DataFrame(vectors)
print(df_vectors.head())

# TSNE은 차원을 축소해서 투사를 해줌
tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

new_values = tsne_model.fit_transform(df_vectors)
df_xy = pd.DataFrame({'words':labels, 'x': new_values[:, 0], 'y': new_values[:, 1]})
print(df_xy.head())

print(df_xy.shape)

df_xy.loc[df_xy.shape[0]] = (key_word, 0,0)
plt.figure(figsize=(8,8))
plt.scatter(0, 0, s=1500, marker='*') # x,y 좌표로 산점도 (위에 'x': new_values 등) / marker 사이즈와 모양

for i in range(len(df_xy.x)):
    a = df_xy.loc[[i, 10], :]
    plt.plot(a.x, a.y, '-D', linewidth=2)
    plt.annotate(df_xy.words[i], xytext=(5, 2), xy=(df_xy.x[i], df_xy.y[i]), textcoords='offset points', ha='right', va='bottom')
    # annotate [도표 상 주석], xytext [출력될 텍스트의 x,y 좌표], xy [실제로 찍을 x,y 좌표], ha [수평 정렬], va [수직 정렬]
    # 파라미터 설명 : https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html
plt.show()