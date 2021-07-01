# 모듈 임포트!
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from konlpy.tag import Okt
import matplotlib as mpl
from matplotlib import font_manager, rc

fontpath = './prj2_Movie_for_you/Jalnan.ttf'
df = pd.read_csv('./prj2_Movie_for_you/crawling/cleaned_review_2019.csv')


