import nltk
import numpy as np
import pandas as pd
import pickle
from dataset.Semantic import Semantic

semantic = Semantic()
df = pd.read_csv('dataset/train.csv')

stemmer = nltk.stem.PorterStemmer()
for i, title in df['produk'].iteritems():
    words = nltk.tokenize.word_tokenize(title.lower())
    for j, word in enumerate(words):
        word = stemmer.stem(word)
        words[j] = word
    words = np.unique(words)
    for word in words:
        semantic.addSem(word)

semantic.calculateIDF(df.shape[0])
with open('dataset/semantic.pkl', 'wb') as f:
    pickle.dump(semantic,f)