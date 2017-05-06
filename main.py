import os
import keras
import numpy as np
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

TEXT_DATA_DIR = 'text_data'
texts = []
l = [] #input
'''
### 处理文本 ####
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    if name[0].isdigit():
        path = os.path.join(TEXT_DATA_DIR, name)
        f = open(path)
        texts.append(f.read())
        f.close()

# word2vec dictionary
embeddings_index = {}
f = open(os.path.join('glove.6B', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    embeddings_index[word] = coefs
f.close()

for text in texts:
    words = keras.preprocessing.text.text_to_word_sequence(text,
        filters= '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=" ")
    num = np.shape(words)[0]
    vector = np.zeros(100)
    for w in words:
        if w in embeddings_index.keys():
            v = embeddings_index[w]
            vector += v
    vector /= num
    vector = np.ndarray.tolist(vector)
    l.append(vector)
'''
### 处理代码 ###
J = 5
CODE_DATA_DIR = 'code_data'
codes = []

codes_vector = []
H = []

c = [] #input

for name in sorted(os.listdir(CODE_DATA_DIR)):
    if name[0].isdigit():
        path = os.path.join(CODE_DATA_DIR, name)
        f = open(path)
        codes.append(f.read())
        f.close()

for code in codes:
    tokens = keras.preprocessing.text.text_to_word_sequence(code,
        filters= '\t\n', lower=True, split=" ")
    v = np.random.normal(size=100) + 1
    codes_vector.append(v)
    h = np.zeros([100, 100])
    for i in range(100):
        h[i][i] = 1/J
    H.append(h)

