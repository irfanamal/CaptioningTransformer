from dataset.Semantic import Semantic
import pickle

with open('dataset/semantic_clean.pkl', 'rb') as f:
    semantic = pickle.load(f)

semantics_list = semantic.getSemantics()
semantics_list = {k: v for k, v in sorted(semantics_list.items(), key=lambda item: item[1], reverse=True)}

with open('dataset/daftar_semantic_clean.txt', 'a+') as f:
    for key in semantics_list:
        f.write('{} {}\n'.format(key, semantics_list[key]))