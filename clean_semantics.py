import string
with open('dataset/daftar_semantic.txt', 'r') as f:
    semantics = f.readlines()

with open('dataset/clean_semantics.txt', 'a+') as f:
    for semantic in semantics:
        semantic = semantic.split()
        term = semantic[0]
        idf = float(semantic[1])
        if not any(char.isdigit() for char in term) and not any(char in string.punctuation for char in term) and len(term)>2:
            f.write('{} {}\n'.format(term, idf))