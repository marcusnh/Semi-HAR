from venv import create
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

from sklearn.datasets import make_classification

# Generates toy dataset for binary classification with shape x = [5000, 20]
def generate_data():
    x, y = make_classification(n_samples=5000, n_features=20, n_classes=2, weights=[0.95, 0.05])
    return x, y

def create_dataset(
    n_samples=1000,
    weights=(0.01, 0.01, 0.98),
    n_classes=3,
    class_sep=0.8,
    n_clusters=1,
):
    return make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=n_clusters,
        weights=list(weights),
        class_sep=class_sep,
        random_state=0,
    )
x,y = create_dataset()
print(x)
x = np.random.rand(3,3,4)
print(x)
# x_new = np.empty((1,4))
# for i,data in enumerate(x):
#     x_new = np.concatenate((x_new,data))
# x_new = np.delete(x_new,0,0)
# print(x_new)
print(x.reshape(-1,4))