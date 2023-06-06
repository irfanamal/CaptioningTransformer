from dataset.Semantic import Semantic
import pickle

with open('dataset/clean_semantics.txt') as f:
    clean_semantics = f.readlines()

semantic = Semantic()
for line in clean_semantics:
    data = line.split()
    term = data[0]
    idf = float(data[1])
    semantic.dictionary[term] = idf

with open('dataset/semantic_clean.pkl', 'wb+') as f:
    pickle.dump(semantic,f)