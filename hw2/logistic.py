# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 04:53:00 2017

@author: kennylin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 21:39:38 2017

@author: kennylin
"""


# coding: utf-8



import numpy as np
import pandas as pd
import sys
import math as floor




def sigm(Z):
    return (1 / (1 + np.exp(-Z)))




def cost_fnt(X, y, w):
    pred = sigm(np.dot(X, w))
    return -np.mean(y * np.log(pred + 1e-10) + (1 - y) * np.log((1 - pred + 1e-10)))




def correctness(X, y, w):
    pred = sigm(np.dot(X, w))

    #print(pred)
    """
    p = pred
    p[pred < 0.5] = 0.0
    p[pred >= 0.5] = 1.0
    """
    p = np.around(pred)
    return np.mean(1 - np.abs(y - p))



def gradient(X, y, w, iteration):
    sum_grad = np.zeros((len(w),1))
    learning_rate = 0.1
    
    for it in range(iteration):
        diff = sigm(np.dot(X, w)) - y
        grad = np.dot(X.T, diff)
        sum_grad += grad ** 2
        w = w - (learning_rate * grad / np.sqrt(sum_grad + 1e-10))
        
        if (it+1) % 500 == 0:
            print("Iteraion = {}, Loss = {}, Correct = {}".format(it, cost_fnt(X, y, w), correctness(X, y, w)))
        
    return w
            
def normalization(X):
    u = np.mean(X , axis = 0)
    std = np.std(X, axis = 0)
    return (X - u) / (std + 1e-10)
        




def read_csv(data):
    df = pd.read_csv(data)
    return df.as_matrix().astype(float)
"""
def fit_degree(X_train, Y_train, X_valid, Y_valid, param):
    degree = 9
    acc = 0
    for i in range(2, degree+1): 
        X_train = np.concatenate((X_train, X_train[:, param] ** i), axis = 1)
        X_valid = np.concatenate((X_valid, X_valid[:, param] ** i), axis = 1)
        X_train_n = normalization(X_train)
        X_valid_n = normalization(X_valid)
        initial_w = np.zeros((X_train_n.shape[1], 1))
        train_w = gradient(X_train_n, Y_train, initial_w, iteration = 5000)
        Ein = correctness(X_train_n, Y_train, train_w)
        Eout = correctness(X_valid_n, Y_valid, train_w)
        print("Degree = {}, Loss = {}, Correct = {}".format(i, cost_fnt(X_train_n, Y_train, train_w), Ein))
        print("Degree = {}, Loss = {}, Correct = {}".format(i, cost_fnt(X_valid_n, Y_valid, train_w), Eout))    
        print((Ein + Eout) / 2)
        if (Ein + Eout) / 2 > acc:
            acc = (Ein + Eout) / 2
            best = i
            w = train_w
    return w, best
"""         

X_train = read_csv(sys.argv[3])
Y_train = read_csv(sys.argv[4])
X_test = read_csv(sys.argv[5])
param = [0, 1, 3, 4, 5]
deg = [2, 3, 4, 5, 6]
valid = 0.1

valid_data_size = int(Y_train.shape[0] * valid)

X_valid, Y_valid = X_train[0:valid_data_size], Y_train[0:valid_data_size]
X_train, Y_train = X_train[valid_data_size:], Y_train[valid_data_size:]

b = np.ones((X_train.shape[0], 1))
for d in deg:
    X_train = np.concatenate((X_train, X_train[:, param] ** d), axis = 1)
X_train = np.concatenate((X_train, b), axis = 1)
X_train = normalization(X_train)
b_valid = np.ones((X_valid.shape[0], 1))
X_valid = np.concatenate((X_valid, b_valid), axis = 1)
#train_w, deg = fit_degree(X_train, Y_train, X_valid, Y_valid, param)
w = np.zeros((X_train.shape[1], 1))
train_w = gradient(X_train, Y_train, w, iteration = 50000)


for d in deg:
    X_test = np.concatenate((X_test, X_test[:, param] ** d), axis = 1)

b_test = np.ones((X_test.shape[0], 1))
X_test = np.concatenate((X_test, b_test), axis = 1)
X_test = normalization(X_test)

with open(sys.argv[6], "w") as f:
    f.write("id,label\n")
    test_y = sigm(np.dot(X_test, train_w))
    test_y = [1 if p >= 0.5 else 0 for p in test_y]
    #print(test_y)
    for id in range(len(test_y)):
        f.write("{},{}\n".format(id+1, test_y[id]))

