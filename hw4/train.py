# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:15:34 2017

@author: kennylin
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, GRU, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from gensim.models import word2vec
from gensim import models
import re
#%%
maxlen = 30
dim = 128

def write(res):
    print('---writing---')
    with open("predict_GRU2.csv", "w") as f:
        f.write("id,label\n")
        for id in range(len(res)):
            ans = np.argmax(res[id])
            f.write("{},{}\n".format(id, ans))

def _model(x_train, y_train, num_words, embedding_matrix ,valid = True, embedding_dim = 128):
    

    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, weights = [embedding_matrix], input_length = maxlen))

    model.add(GRU(128, dropout = 0.5))
    
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation = 'sigmoid'))

    model.summary()
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    if valid:
        history = model.fit(x_train, y_train, batch_size = 256, epochs = 2, validation_split = 0.05)
    else:
        history = model.fit(x_train, y_train, batch_size = 128, epochs = 2)

    #score = model.evaluate(x_train, y_train)
    #result = model.predict(x_test)
    
    #write(result)
    model.save('ver2.6_GRU.h5')
    #print('Test loss:', score[0])
    #print('Test accuracy:', score[1])
        
#%%
print('---read data---')
with open(sys.argv[1], 'r', encoding = 'utf-8') as f:
    labels = []
    sentences = []
    s = []
    for l in f:
        l = l.replace('+++$+++', '').split(" ", 1)
        labels.append(l[0])
        sentences.append(l[1])
        s.append(l[1].split())
with open(sys.argv[2], 'r', encoding = 'utf-8') as f:
    nolabel_sentences = []
    s3 = []
    for l_nolabel in f:
        nolabel_sentences.append(l_nolabel)
        s3.append(l_nolabel.split())

#%%
print('word2vec') 
model_w = models.Word2Vec.load(sys.argv[3])
print('---tokenizer---')
with open(sys.argv[4], 'rb') as handle:
    token = pickle.load(handle)
train_seq = token.texts_to_sequences(sentences)
x_train = pad_sequences(train_seq, maxlen = maxlen)
#y_train =np.array(labels)
y_train = np_utils.to_categorical(labels, 2)
num_words = len(token.word_index) + 1

#%%
print('---embedding---')
embedding_matrix = np.zeros((num_words, dim))
for word, vector in token.word_index.items():
    if word in model_w.wv:
        embedding_matrix[vector] = model_w.wv[word] 
#%%
print('---train---')
_model(x_train, y_train, num_words, embedding_matrix)
