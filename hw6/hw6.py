# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 21:06:41 2018

@author: kennylin
"""

import csv
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import load_model
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam

def normalization(X):
    u = np.mean(X , axis = 0)
    std = np.std(X, axis = 0)
    return (X - u) / (std + 1e-10)
print('load...')
data = np.load(sys.argv[1])

print("normalize...")
n_data = normalization(data)
"""
encoding_dim = 32
input_img = Input(shape = (784,))

encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(48, activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

decoded = Dense(64, activation='relu')(encoder_output)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

encoder = Model(input=input_img, output=encoder_output)

autoencoder = Model(input=input_img, output=decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(n_data, n_data, epochs=300, batch_size=512, shuffle=True, validation_split = 0.1)
encoder.save("hw6.h5")
"""
#%%
"""
print("pca...")
pca = PCA(n_components = 32, whiten = True).fit(n_data)
x = pca.transform(n_data)
"""
#%%
print('kmeans...')
encoder = load_model("hw6.h5")
x = encoder.predict(n_data)
k = KMeans(n_clusters = 2).fit(x)

#%%
count = 0
for i in range(140000):
    if k.labels_[i] == 0:
        count += 1

#%%

with open(sys.argv[2]) as f:
    f.readline()
    image1 = []
    image2 = []
    for row in csv.reader(f):
        image1.append(int(row[1]))
        image2.append(int(row[2]))
        
with open(sys.argv[3], "w") as f:
    f.write("ID,Ans\n")
    for i in range(1980000):
        if k.labels_[image1[i]] == k.labels_[image2[i]]:
            pic = 1
        else:
            pic = 0
        f.write("{},{}\n".format(i, pic))

