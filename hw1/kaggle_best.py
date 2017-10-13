# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 20:32:43 2017

@author: kennylin
"""
import numpy as np
import pandas as pd
import sys

iteraion_time = 30
def cost_fnt(X, y, w):
    N = len(y)
    error = (1/N) * sum((np.dot(X, w) - y)**2)
    return error

def gradient_descent(X, y, w):
    X = np.array(X, dtype='float')
    w = np.dot(np.linalg.pinv(X), y)
    X = np.array(X, dtype='object')
    
    return w

df = pd.read_csv('train.csv', encoding = 'big5')
df_test = pd.read_csv(sys.argv[1], encoding = 'big5', header = None)
raw_data = df.as_matrix()
raw_test_data = df_test.as_matrix()

learning_rate = 0.1
convergence_bound = 0.000001
validation_rate = 60
weight_num = 6 * 9 + 1
train_parameter_num = 3

train_PM25 = []
train_PM25_2 =[]
train_var = []
train_var_2 = []
count = 0
cut = True
for i in range(0, 4320):
    count += 1
    if cut:
        if (i - 10) % 18 == 0:
            raw_data[i, 3:] = [rain.replace("NR", "0") for rain in raw_data[i, 3:]]
        raw_data[i, 3:] = [float(data) for data in raw_data[i, 3:]]
        if i % 18 == 9:
            train_PM25_2.append(raw_data[i, 3:])
            train_var_2.append(raw_data[i, 3:])
        if i % 18 == 7 or i % 18 == 8:
            train_var_2.append(raw_data[i, 3:])
    
    else:
        if (i - 10) % 18 == 0:
            raw_data[i, 3:] = [rain.replace("NR", "0") for rain in raw_data[i, 3:]]
        raw_data[i, 3:] = [float(data) for data in raw_data[i, 3:]]
        if i % 18 == 9:
            train_PM25.append(raw_data[i, 3:])
            train_var.append(raw_data[i, 3:])
        if i % 18 == 7 or i % 18 == 8:
            train_var.append(raw_data[i, 3:])
    
    if count == 20 * 18:
        cut = False
    if count == 60 * 18:
        cut = True
        count = 0
    

test_var = []
for i in range(0, 4320):
    if (i - 10) % 18 == 0:
        raw_test_data[i, 2:] = [rain.replace("NR", "0") for rain in raw_test_data[i, 2:]]
    raw_test_data[i, 2:] = [float(data) for data in raw_test_data[i, 2:]]

    if i % 18 == 9 or i % 18 == 8 or i % 18 == 7:
       test_var.append(raw_test_data[i, 2:])

test_var = np.array(test_var)
train_var = np.array(train_var)
train_PM25 = np.array(train_PM25)
train_var_2 = np.array(train_var_2)
train_PM25_2 = np.array(train_PM25_2)

w = np.zeros((weight_num, 1))
w_2 = np.zeros((weight_num,1))
temp = train_var[0:train_parameter_num]
temp_2 = train_var_2[0:train_parameter_num]

for i in range(train_parameter_num, train_parameter_num * 160, train_parameter_num):
   temp = np.concatenate((temp, train_var[i:i+train_parameter_num]), axis = 1)
   
for i in range(train_parameter_num, train_parameter_num * 80, train_parameter_num):
   temp_2 = np.concatenate((temp_2, train_var_2[i:i+train_parameter_num]), axis = 1)
   
train_var = temp
train_var_2 = temp_2

squared_num = 3
for i in range(squared_num):
    train_var = np.concatenate((train_var, np.square(train_var[i]).reshape(1,-1)), axis = 0)
for i in range(squared_num):
    train_var_2 = np.concatenate((train_var_2, np.square(train_var_2[i]).reshape(1,-1)), axis = 0)

    
train_PM25 = train_PM25.reshape(-1,1)[9:]
train_PM25_2 = train_PM25_2.reshape(-1,1)[9:]

train_X = train_var[:, 0:9].reshape(1,-1)
train_X_2 = train_var_2[:, 0:9].reshape(1,-1)

for j in range(1, 3831, 1):
    train_X= np.concatenate((train_X, train_var[:, j:j+9].reshape(1,-1)), axis = 0)
for k in range(1, 1911, 1):
    train_X_2 = np.concatenate((train_X_2, train_var_2[:, k:k+9].reshape(1,-1)), axis = 0)
    
b = np.ones((3831,1))
b_2 = np.ones((1911,1))
train_X = np.concatenate((train_X, b), axis = 1)
train_X_2 = np.concatenate((train_X_2, b_2), axis = 1)
w = gradient_descent(train_X, train_PM25, w)
w_2 = gradient_descent(train_X_2, train_PM25_2, w_2)
#print(cost_fnt(train_X_2, train_PM25_2, w) + cost_fnt(train_X, train_PM25, w))

#print(cost_fnt(train_X, train_PM25, w_2) + cost_fnt(train_X_2, train_PM25_2, w_2))

with open(sys.argv[2], "w") as f:
    f.write("id,value\n")
    for id in range(240):
        test_x = test_var[id*3:(id+1) * 3]
        
        for i in range(squared_num):
            test_x = np.concatenate((test_x, np.square(test_x[i]).reshape(1,-1)), axis = 0)
        
        test_x = np.reshape(test_x, (1,-1))
        test_x = np.concatenate((test_x, np.ones((1,1))), axis = 1)
        
        test_y = np.dot(test_x, w)[0][0]
        f.write("id_" + str(id) + "," + str(test_y) + "\n")
