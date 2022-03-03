from enum import unique
import os
import shutil
import numpy as np
from collections import  Counter 
import seaborn as sns

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import MultiLabelBinarizer

'''
DESCRIPTION OF FILE: TODO
'''

'''
Helper functions for verifiing solution
'''
def remove_folder(path='wandb'):
    try:
        shutil.rmtree(path)
        print(f'Folder {path} was removed')
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    return 

def count_nr_classes(y_data, classes=3, verbose=1):
    '''
    Count classes for one-hot-encoding class data
    '''
    nr_values = np.zeros(classes)
    for i in y_data:
        nr_values[np.argmax(i)] +=1
  
    if verbose >0:
        print("The class distribution of the data are:")
        for i, nr in enumerate(nr_values):
            print(f'Class {i+1}: {int(nr)}')
            

def convert_one_hot_encoding(y_one_hot_e, verbose =1):
    y_int = np.argmax(y_one_hot_e, axis=1)
    if verbose >0:
        unique, counter = np.unique(y_int, return_counts=True)
        print("The class distribution of the data are:")
        for i, nr in zip(unique, counter):
            print(f'Class {i+1}: {nr}')

    return y_int

'''
Over and under-sampling section
'''
def under_sample(x,y):
    # Applies random undersampling
    rus = RandomOverSampler()
    x,y =rus.fit_resample(x,y)
    return x,y

def plot_resampling(X, y, sampler, ax, title=None):
    X_res, y_res = sampler.fit_resample(X, y)
    ax.scatter(X_res[:, 0], X_res[:, 1], c=y_res, alpha=0.8, edgecolor="k")
    if title is None:
        title = f"Resampling with {sampler.__class__.__name__}"
    ax.set_title(title)
    sns.despine(ax=ax, offset=10)

'''
Class weights for training models
'''

def class_weights(activity_count, tot_size):
    class_weights = {}
    for c,count in activity_count.items():
        print(c,count)
        class_weights[c] = tot_size/(count*len(activity_count))
    print(class_weights)

def get_weights(y_train):
    # For one-hot-encoded training data
    y_integer = np.argmax(y_train, axis=1)
    y_labels = np.unique(y_integer)
    class_weights = compute_class_weight(class_weight='balanced',
                                        classes=y_labels,
                                        y=y_integer)
    print(dict(zip(y_labels, class_weights)))
    return dict(zip(y_labels, class_weights))