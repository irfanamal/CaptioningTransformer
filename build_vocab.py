import pandas as pd
import pickle
from dataset.Vocabulary import Vocabulary

vocab = Vocabulary()
df = pd.read_csv('dataset/train.csv')

vocab.addWord('<PAD>')
vocab.addWord('<S>')
vocab.addWord('</S>')
vocab.addWord('<UNK>')

for i,data in df.iterrows():
    title = data['produk'].lower().split()
    for word in title:
        vocab.addWord(word)

with open('dataset/vocab.pkl', 'wb') as f:
    pickle.dump(vocab,f)