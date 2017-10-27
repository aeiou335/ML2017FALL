# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:37:06 2017

@author: kennylin
"""

import numpy as np
import pandas as pd
import sys
import math

def sigm(Z):
    return 1/(1 + np.exp(Z))

def cal_mean_cov(c1, c2):
    mu_1 = np.mean(c1, axis = 0)
    mu_2 = np.mean(c2, axis = 0)
    cov_1 = cov(c1, mu_1)
    cov_2 = cov(c2, mu_2)
    num_1 = c1.shape[0]
    num_2 = c2.shape[0]
    num = num_1 + num_2
    covarience = (num_1 / num) * cov_1 + (num_2 / num) * cov_2
    
    return mu_1, mu_2, covarience, num_1, num_2
    
def cov(x, mu):
    return np.mean([(x[i]-mu).reshape(-1,1) * (x[i]-mu).reshape(1,-1) for i in range(x.shape[0])], axis = 0)

def generative(X, Y):
    
    C_1 = X[(Y == 0).reshape(-1,)]
    C_2 = X[(Y == 1).reshape(-1,)]
    mu_1, mu_2, cov, N_1, N_2 = cal_mean_cov(C_1, C_2)
   
    cov_inv = np.linalg.inv(cov)
    w = np.dot((mu_1 - mu_2), cov_inv)
    b = (-1/2) * np.dot(np.dot([mu_1], cov_inv), mu_1) + (1/2) * np.dot(np.dot([mu_2], cov_inv), mu_2) + np.log(float(N_1/N_2))
    return w, b

def normalization(X):
    u = np.mean(X , axis = 0)
    std = np.std(X, axis = 0)
    return (X - u) / (std + 1e-10)

def read_csv(data):
    df = pd.read_csv(data)
    return df.as_matrix().astype(float)

def main(argv):
    X_train = read_csv(argv[3])   
    Y_train = read_csv(argv[4])
    X_test = read_csv(argv[5])
    """
    X_train = normalization(X_train)
    X_test = normalization(X_test)
    """
    train_w, train_b = generative(X_train, Y_train)
    Y_pred = sigm(np.dot(train_w, X_train.T) + train_b)
    Y = np.around(Y_pred)
    acc = np.mean(Y_train.flatten() == Y)
    print("Accuracy:{}".format(acc))
    
    
    with open(argv[6], "w") as f:
        f.write("id,label\n")
        test_y = sigm(np.dot(train_w, X_test.T) + train_b)
        test_y = [1 if p >= 0.5 else 0 for p in test_y]
        for id in range(len(test_y)):
            f.write("{},{}\n".format(id+1, test_y[id]))
    
if __name__ == '__main__':
    main(sys.argv)
