import os
import keras
import collections
import numpy as np
import tensorflow as tf
import math
from keras.engine import Input
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.text import text_to_word_sequence
from tensorflow.contrib.keras.python.keras.engine import Model


# load the data
TEXT_DATA_DIR = 'text_data'
CODE_DATA_DIR = 'code_data'
texts = []
codes = []
for name in sorted(os.listdir(TEXT_DATA_DIR)):
    if name[0].isdigit():
        path = os.path.join(TEXT_DATA_DIR, name)
        path2 = os.path.join(CODE_DATA_DIR, name)
        f = open(path)
        f2 = open(path2)
        text_tokens = keras.preprocessing.text.text_to_word_sequence(f.read(),
          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ")
        code_tokens = keras.preprocessing.text.text_to_word_sequence(f2.read(),
                filters='\t\n', lower=True, split=" ")
        texts.append(text_tokens)
        codes.append(code_tokens)
        f.close()
        f2.close()

print('data loaded!')
print(texts)
print(codes)

# get input label
input_text = []
input_code = []
label = []
WINDOW = 10
for i in range(np.shape(texts)[0]):
    text = texts[i]
    code = codes[i]
    n = np.shape(code)[0]
    for j in range(WINDOW, n-1):
        input_code.append(code[j-WINDOW: j])
        input_text.append(text)
        label.append(code[j+1])

number = np.shape(input_text)[0]  # train data size
print('raw data processed !')
for i in range(number):
    print(input_text[i])
    print(input_code[i])
    print(label[i])


# build word dictionary
word_dictionary = {}
f = open(os.path.join('glove.6B', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype = 'float32')
    word_dictionary[word] = coefs
f.close()


# code dictionary
all_tokens = []
for tokens in codes:
    all_tokens.extend(tokens)

def build_dataset(all_tokens, input_list):
  dictionary = dict()
  for token in all_tokens:
      dictionary[token] = len(dictionary)

  data_list = []
  for i in range(np.shape(input_list)[0]):
      data = []
      for token in input_list[i]:
          index = dictionary[token]
          data.append(index)
      data_list.append(data)
  return data_list, dictionary

input_code, dictionary = build_dataset(all_tokens, input_code)



# initializations
embedding_size = 100  # vector dimension
num_sampled = 64    # Number of negative examples to sample.
tokens_size = len(dictionary)  # number of code tokens

train_inputs1 = tf.placeholder(tf.float32, shape=(embedding_size,))  # text
train_inputs2 = tf.placeholder(tf.int32, shape=(WINDOW,))  # code
train_labels = tf.placeholder(tf.float32, shape=(embedding_size,))

embeddings = tf.Variable(tf.random_uniform([tokens_size, embedding_size], -1.0, 1.0)) # code
embed = tf.nn.embedding_lookup(embeddings, train_inputs2)



# get H matrix variable
h_temp = np.zeros([100, 100])
for j in range(100):
    h_temp[j][j] = 1 / 10

H = []
for i in range(10):
    h = tf.Variable(h_temp)
    H.append(h)

x_code = []
for code in input_code:
    vector = np.zeros(embedding_size)
    for i in range(WINDOW):
        c = code[i]
        if c in token_embeddings.keys():
            v = np.dot(H[i], token_embeddings[c])
            print(v)
            vector += v
    vector = np.ndarray.tolist(vector)
    x_code.append(vector)


# get input data
x_text = []
for text in input_text:
    num = np.shape(text)[0]
    vector = np.zeros(100)
    for t in text:
        if t in word_dictionary.keys():
            v = word_dictionary[t]
            vector += v
    vector /= num
    vector = np.ndarray.tolist(vector)
    x_text.append(vector)

x_data = x_text * x_code

# get output data
y_data = []
for lab in label:
    y_data.append(token_embeddings[lab])


nce_weights = tf.Variable(
    tf.truncated_normal([number, embedding_size],
                        stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([number]))

loss = tf.reduce_mean(
      tf.nn.nce_loss(weights=nce_weights,
                     biases=nce_biases,
                     labels=train_labels,
                     inputs=train_inputs,
                     num_sampled=num_sampled,
                     num_classes=number))

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Add variable initializer.
init = tf.global_variables_initializer()


# begin training
num_steps = 100001

init.run()
print('Initialized')

average_loss = 0
for step in range(num_steps):
    feed_dict = {train_inputs: x_data, train_labels: y_data}