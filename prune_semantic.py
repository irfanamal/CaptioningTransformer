from dataset.Semantic import Semantic
import pickle

with open('dataset/semantic_clean.pkl', 'rb') as f:
    semantic = pickle.load(f)

semantic.trim(6)
with open('dataset/semantic6.pkl', 'wb') as f:
    pickle.dump(semantic,f)