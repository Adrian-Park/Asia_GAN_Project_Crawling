# 모듈 임포트!
import pandas as pd
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from matplotlib import font_manager, rc
import matplotlib as mpl

# 폰트 설정
font_path = './Jalnan.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus']=False
rc('font', family=font_name)

embedding_model = Word2Vec.load('./models/word2VecModel_2016_2021.model')
key_word = '라이온킹'
sim_word = embedding_model.wv.most_similar(key_word, topn=10) # 모델을 이용하여 코사인 값이 가장 유사한 단어 10개 추출
print(sim_word)