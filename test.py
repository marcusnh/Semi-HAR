import numpy as np
import os
import sys
import shutil
# from sklearn.utils import shuffle

# a = np.arange(18)
# a = a.reshape(2,3,3)
# b = np.arange(20,56).reshape(4,3,3)
# y1 = np.ones(18).reshape(2,3,3)
# y2 = np.zeros(36).reshape(4,3,3)
# y = np.concatenate((y1,y2))

# c = np.concatenate((a,b))
# print(c)
# print(y)
# print('shuffling')
# shuffle = np.random.permutation(len(y))
# d = c[shuffle]
# y = y[shuffle]
# print(shuffle)
# # print(d)
# # print(d.shape)
# # print(y)
# # print(y.shape)


# a = np.arange(10)
# y = np.arange(0,1,0.05)
# yk = y>=0.85
# y_masked = np.where(yk, y, 0)
# print(y_masked)
# print(np.argpartition(-y_masked,0))
# c = np.full(20,False, dtype=bool)
# print(yk | c)
# b = (np.argmax(a,axis=0) <=5) & c
# # print(b)

