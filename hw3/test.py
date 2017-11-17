# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:51:21 2017

@author: kennylin
"""
import sys
import csv
import numpy as np
from keras.models import load_model

def normalization(X):
    u = np.mean(X , axis = 0)
    std = np.std(X, axis = 0)
    return (X - u) / (std + 1e-10)
X_test = []
print('open data...')
with open(sys.argv[1], 'r') as f2:
    for row in list(csv.reader(f2))[1:]:
        for data in row[1].split():
            X_test.append(float(data))
print('wait...')
X_test = np.array(X_test)
X_test = X_test / 255
X_test = X_test.reshape(-1, 48, 48, 1)
X_test = normalization(X_test)
model1 = load_model(sys.argv[3])

print('predict data...')
res = model1.predict(X_test)

with open(sys.argv[2], "w") as f:
    f.write("id,label\n")
    for id in range(len(res)):
        ans = np.argmax(res[id])
        f.write("{},{}\n".format(id, ans))