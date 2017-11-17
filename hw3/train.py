# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 23:20:08 2017

@author: kennylin
"""
import sys
import csv
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator

#%%
"""
def write(res):
    with open("predict.csv", "w") as f:
        f.write("id,label\n")
        for id in range(len(res)):
            ans = np.argmax(res[id])
            f.write("{},{}\n".format(id, ans))
"""
#%%
def _model(x_train, y_train, validation, num_classes):
    model = Sequential()
    
    model.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size = (3, 3), activation='relu'))
    model.add(BatchNormalization())    
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    model.summary()
    
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    datagen = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
    
    if validation:
        x_valid, x_train = x_train[0:2000], x_train[2000:]
        y_valid, y_train = y_train[0:2000], y_train[2000:]
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 72), steps_per_epoch = len(X_train) // 36, 
                                  epochs = 100, validation_data = (X_valid, Y_valid))
    else:
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size = 72), steps_per_epoch = len(X_train) // 72, 
                                  epochs = 60)
    score = model.evaluate(x_train, y_train)
    result = model.predict(X_test)
    model.save(sys.argv[2])
    #write(result)    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
#%%
def normalization(X):
    u = np.mean(X , axis = 0)
    std = np.std(X, axis = 0)
    return (X - u) / (std + 1e-10)
"""
def load_data(data):
    df = pd.read_csv(data)

    

def main():
    df = pd.read_csv('train.csv')  
    

    
if __name__ == '__main__':
    main()
"""
#df = pd.read_csv(sys.argv[1])
"""
df = pd.read_csv('train.csv')
df2 = pd.read_csv('test.csv')
"""
#%%
X = []
X_test = []
Y = []
"""
with open('train.csv', 'r') as f:
    for row in list(csv.reader(f))[1:]:
        Y.append(float(row[0]))
        for data in row[1].split():
            X.append(float(data))

with open('test.csv', 'r') as f2:
    for row in list(csv.reader(f2))[1:]:
        for data in row[1].split():
            X_test.append(float(data))
"""
with open(sys.argv[1], 'r') as f:
    for row in list(csv.reader(f))[1:]:
        Y.append(float(row[0]))
        for data in row[1].split():
            X.append(float(data))

#%%
Y_train = np_utils.to_categorical(Y, 7)
X_train = np.array(X)
X_train = X_train / 255
X_train = X_train.reshape(-1, 48, 48, 1)
X_train = normalization(X_train)


#%%

_model(X_train, Y_train, True, 7)    
