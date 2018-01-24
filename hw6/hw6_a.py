# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 12:00:05 2018

@author: kennylin
"""
import sys
import numpy as np
from skimage import io, filters, transform
import matplotlib.pyplot as plt
from numpy.linalg import svd, eig

data = []

for i in range(415):
    data.append(io.imread(sys.argv[1]+"/{}.jpg".format(i)))

data = np.array(data)
average = np.mean(data, axis = 0)
print(average.shape)

average -= np.min(average)
average /= np.max(average)
average = (average * 255).astype(np.uint8)
io.imsave("test.jpg",average)

img = []
"""
for i in range(415):
	new_img = transform.resize(data[i], (128,128))
	new_img = (new_img * 255).astype(np.uint8)
	img.append(new_img.flatten())
"""
x = np.array(data).reshape(415,600*600*3)
#print(img.shape)
#x = img.flatten()
print(x.shape)
#io.imsave("test2.jpg", new_img)

x_mean = np.mean(x, axis = 0)
U, s, V = svd(x - x_mean, full_matrices=False)

eigenfaces = V
print(s.shape)
print(U.shape)
print(V.shape)

#%%
pic = int(sys.argv[2].split('.')[0])
r = x[pic]
construct = r - x_mean
combine = 0
for i in range(4):
    combine += np.dot(V[i], construct) * V[i]
combine += x_mean    
combine -= np.min(combine)
combine /= np.max(combine)
combine = (combine * 255).astype(np.uint8).reshape(600,600,3)
    
io.imsave("reconstruct.jpg".format(r),combine)
#%%
for i in range(4):
    print(s[i]/np.sum(s))
