# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:50:11 2017

@author: kennylin
"""

import sys
import csv
import numpy as np
from keras.models import load_model
import keras.backend as K
def rmse(y_true, y_pred): 
    return K.sqrt( K.mean((y_pred - y_true)**2) )
def write(res):
    print('---writing---')
    with open(sys.argv[2] , "w") as f:
        f.write("TestDataID,Rating\n")
        for id in range(len(res)):
            #ans = np.argmax(res[id])
            f.write("{},{}\n".format(id+1, res[id][0]))
with open(sys.argv[1], 'r') as f:
    f.readline()
    test_user = []
    test_movie = []
    for row in csv.reader(f):
        test_user.append(float(row[1]))
        test_movie.append(float(row[2]))
test_user = np.array(test_user)
test_movie = np.array(test_movie)

model = load_model(sys.argv[5], custom_objects={'rmse': rmse})

res = model.predict([test_user, test_movie])
mu = 3.58171208604
std = 1.11689766115
res = res * std + mu
res = np.clip(res,1,5)
write(res)