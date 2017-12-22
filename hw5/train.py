# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 02:47:13 2017

@author: kennylin
"""

import numpy as np
import pandas as pd

from keras.preprocessing import sequence
from keras.models import Sequential , Model
from keras.layers import Input, Add, Embedding, Flatten, Dot, Dropout
from keras.layers import Dense , RepeatVector
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.losses import hinge
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K

import sys
import csv
#%%
def write(res):
    print('---writing---')
    with open("predict2.csv", "w") as f:
        f.write("TestDataID,Rating\n")
        for id in range(len(res)):
            #ans = np.argmax(res[id])
            f.write("{},{}\n".format(id+1, res[id][0]))
            
def rmse(y_true, y_pred): 
    return K.sqrt( K.mean((y_pred - y_true)**2) )
#%%
#read data
print('---read data---')
with open('train.csv', 'r') as f:
    f.readline()
    user = []
    movie = []
    rating = []
    for row in csv.reader(f):
        user.append(float(row[1]))
        movie.append(float(row[2]))
        rating.append(float(row[3]))

with open('test.csv', 'r') as f:
    f.readline()
    test_user = []
    test_movie = []
    for row in csv.reader(f):
        test_user.append(float(row[1]))
        test_movie.append(float(row[2]))

total_users = int(max(user))
total_movies = int(max(movie))
user = np.array(user)
movie = np.array(movie)
rating = np.array(rating)
mu = np.mean(rating)
std = np.std(rating)
rating = (rating - mu) / std
dim = 128
print('---build model---')
input_users = Input(shape = (1,))
input_movies = Input(shape = (1,))

embedding_users = Embedding(total_users, dim)(input_users)
embedding_movies = Embedding(total_movies, dim)(input_movies)

embedding_users = Flatten()(embedding_users)
embedding_users = Dropout(0.3)(embedding_users)
embedding_movies = Flatten()(embedding_movies)
embedding_movies = Dropout(0.3)(embedding_movies)

dot = Dot(axes = 1)([embedding_users, embedding_movies])

print('---add bias---')
bias_user = Flatten()(Embedding(total_users, 1)(input_users))
bias_movie = Flatten()(Embedding(total_movies, 1)(input_movies))

out = Add()([dot, bias_user, bias_movie])

#out = dot
model = Model(inputs = [input_users, input_movies], output = out) 
model.summary()

callbacks = []
callbacks.append(EarlyStopping(monitor='val_rmse', patience=3))
callbacks.append(ModelCheckpoint('hw5.h5', save_best_only=True, monitor='val_rmse'))

idx = np.random.permutation(len(user))
print(idx)
user, movie, rating = user[idx], movie[idx], rating[idx]

model.compile(loss = 'mse', optimizer = 'adam', metrics = [rmse])
model.fit([user, movie], rating, batch_size = 1024, epochs = 6, callbacks = callbacks, validation_split = 0.1)

test_user = np.array(test_user)
test_movie = np.array(test_movie)
result = model.predict([test_user, test_movie])
result = result * std + mu
result = np.clip(result,1,5)

print('---done---')
#write(result)