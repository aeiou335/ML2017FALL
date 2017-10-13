# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 02:18:39 2017

@author: kennylin
"""
import numpy as np
import pandas as pd
import math
import sys

iteraion_time = 15000
def cost_fnt(X, y, w):
    N = len(y)
    error = (1/N) * sum((np.dot(X, w) - y)**2)
    return error

def gradient_descent(X, y, w):
    n = len(y)
    gradients = 0
    errors = [cost_fnt(X, y, w)[0]]
    for iteration in range(iteraion_time):
        diff = np.dot(X, w) - y
        #print(diff.shape)
        gradient = (2/n) * np.dot(X.T ,diff)
        #print (gradient.shape)
        #gradient = np.reshape(gradient, (len(gradient), 1))
        gradients += sum(np.square(gradient))
            #print(gradient.shape)
        w = w - (learning_rate * gradient / math.sqrt(gradients))
        errors.append(cost_fnt(X, y, w)[0])
        error_diff = errors[iteration] - errors[iteration + 1]
        #print(trainw)
      
        if error_diff < 0:
            print('Gradient descent overshooting with error difference of ' + str(error_diff) + ' and error of ' +
                  str(errors[iteration]))
            
        if abs(error_diff) < convergence_bound:
            print('Gradient descent converged after ' + str(iteration) + ' iterations with error difference of '
                      + str(error_diff) + ' and error of ' + str(errors[iteration]))
            break
           
        print(math.sqrt(errors[iteration]))
        if iteration % 100 == 0:
            print(iteration)
    return w

df = pd.read_csv('train.csv', encoding = 'big5')
df_test = pd.read_csv(sys.argv[1], encoding = 'big5', header = None)
raw_data = df.as_matrix()
raw_test_data = df_test.as_matrix()

learning_rate = 0.1
convergence_bound = 0.000001
validation_rate = 60
weight_num = 3 * 9 + 1
train_parameter_num = 8

train_PM25 = []
train_var = []
validation_PM25 = []
validation_var = []
#np.set_printoptions(precision = 3, suppress = True)

for i in range(0, 4320):
    if (i - 10) % 18 == 0:
        raw_data[i, 3:] = [rain.replace("NR", "0") for rain in raw_data[i, 3:]]
    raw_data[i, 3:] = [float(data) for data in raw_data[i, 3:]]
    if i % 18 == 9:
        train_PM25.append(raw_data[i, 3:])
        train_var.append(raw_data[i, 3:])
    
    if i % 18 == 7 or i % 18 == 8:
       train_var.append(raw_data[i, 3:])
    """
    if i % 18 != 10 and i % 18 != 16:
        train_var.append(raw_data[i, 3:])
    """
#initiate train set and validation set
"""
for i in range(0, 4320):
    if (i - 10) % 18 == 0:
        raw_data[i, 3:] = [rain.replace("NR", "0") for rain in raw_data[i, 3:]]
    raw_data[i, 3:] = [float(data) for data in raw_data[i, 3:]]
    if i % 18 == 9:
        train_PM25.append(raw_data[i, 3:])
        train_var.append(raw_data[i, 3:])
    else:
        train_var.append(raw_data[i, 3:])
"""
test_var = []
for i in range(0, 4320):
    if (i - 10) % 18 == 0:
        raw_test_data[i, 2:] = [rain.replace("NR", "0") for rain in raw_test_data[i, 2:]]
    raw_test_data[i, 2:] = [float(data) for data in raw_test_data[i, 2:]]
    #test_var.append(raw_test_data[i, 2:])
    
    if i % 18 == 9 or i % 18 == 8 or i % 18 == 7:
       test_var.append(raw_test_data[i, 2:])
    """
    if i % 18 != 10 and i % 18 != 16:
        test_var.append(raw_test_data[i, 2:])
    """
test_var = np.array(test_var)
train_var = np.array(train_var)
train_PM25 = np.array(train_PM25)
w = np.zeros((weight_num, 1))
temp = train_var[0:3]
for i in range(3, 3 * 240, 3):
   temp = np.concatenate((temp, train_var[i:i+3]), axis = 1)
train_var = temp

"""
for i in range(3):
    train_var = np.concatenate((train_var, np.square(train_var[i]).reshape(1,-1)), axis = 0)
"""
    
train_PM25 = train_PM25.reshape(-1,1)[9:]
"""
train_X = train_var[:, 0:9].reshape(1,-1)
train_y = train_PM25[0]
w = gradient_descent(train_X, train_y, w)
#print(train_X)
"""
train_X = train_var[:, 0:9].reshape(1,-1)
for j in range(1,5751,1):
    train_X= np.concatenate((train_X, train_var[:, j:j+9].reshape(1,-1)), axis = 0)
b = np.ones((5751,1))
train_X = np.concatenate((train_X, b), axis = 1)
#w = gradient_descent(train_X, train_PM25, w)

w = np.array([[-0.01896264773570659],
       [0.0185130225290205],
       [-0.006060065382868092],
       [-0.006659919394645992],
       [0.005684259562206006],
       [-0.02177059045311029],
       [-0.001326315391062937],
       [-0.014728433071091403],
       [0.0661494977903264],
       [0.002334858473302979],
       [0.0020768597141502376],
       [-0.016702549489876158],
       [0.030298384887420312],
       [-0.019528910589718656],
       [-0.015665945338637664],
       [0.02248817865827605],
       [-0.020160091855405977],
       [0.07847823927789929],
       [-0.03125524315291587],
       [0.013892451103704375],
       [0.1540224402827776],
       [-0.19346605179837797],
       [-0.00713981459980143],
       [0.4285181302942966],
       [-0.5223212349689508],
       [0.019793642704785765],
       [0.9774673004634239],
       [0.010703100760387515]], dtype=object)

with open(sys.argv[2], "w") as f:
    f.write("id,value\n")
    for id in range(240):
        test_x = test_var[id*3:(id+1) * 3]

        test_x = np.reshape(test_x, (1,-1))
        test_x = np.concatenate((test_x, np.ones((1,1))), axis = 1)
        
        test_y = np.dot(test_x, w)[0][0]
        f.write("id_" + str(id) + "," + str(test_y) + "\n")

#print(type((raw_data[1, 0])))
