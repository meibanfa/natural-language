import word2vec
import numpy as np

## description ##
def getVector(filename):
    # train
    word2vec.word2phrase(filename, 'text-phrase', verbose=True)
    word2vec.word2vec('text-phrase', 'text.bin', size = 100, verbose = True)

    # prediction
    model = word2vec.load('text.bin', encoding="ISO-8859-1")

    (m, n) = model.vectors.shape
    temp = np.ones([1, m])
    result = np.dot(temp, model.vectors)
    return result

## input
l1 = getVector('nl1.txt')