# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:46:58 2022

@author: Stian
"""

# Compare loss functions MCC and CCE

import matplotlib.pyplot as plt
import numpy as np
#from cv2 import Sobel, Laplacian, watershed
#import cv2 as cv
import seaborn as sns

import json

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mycolorpy import colorlist as mcp

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow_addons.metrics import F1Score

import tensorflow as tf
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input 

from models.vanilla_unet import vanilla_unet


# =============================================================================
# Load data
# =============================================================================
def load_data(path="data/roofs/"):
    
    X_train = np.load(path+"X_data.npy") 
    y_train = np.load(path+"y_data.npy")
    
    X_val = np.load(path+"X_data_val.npy")
    y_val = np.load(path+"y_data_val.npy")
    
    X_test = np.load(path+"X_data_test.npy")
    y_test = np.load(path+"y_data_test.npy")

    
    
    y_train = redesign_y(y_train)
    y_val = redesign_y(y_val)
    y_test = redesign_y(y_test)

    # print(y_train.shape)
    # print(y_val.shape)
    # print(y_test.shape)
    
    return X_train, y_train, \
           X_val, y_val, \
           X_test, y_test


def redesign_y(y):
  n,r1,c1,d = y.shape
  # Adds a new dimension of layer too have two class problem.
  yy = np.append(y, np.zeros((n, r1, c1,d)), axis=3)
  for i in range(int(y.max()-1)):  
    yy = np.append(yy, np.zeros((n, r1, c1,d)), axis=3)
  #yy[yy >= 0.001] = 1
  yy1 = yy.copy()
  yy1[:,:,:,0] = 0 # reset map
  for i in range(n):
    values = yy[i,:,:,0]
    for r in range(r1):
      for c in range(c1):
        value = yy[i,r,c,0]
        yy1[i,r,c,int(value)] = 1

  return yy1

path="data/roofs/"
X_train, y_train, X_val, y_val, X_test, y_test = load_data(path)


train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

train_ds = (
    train_dataset
    #.shuffle(1000)
    #.map(f, num_parallel_calls=AUTOTUNE)
    .batch(32)
    #.prefetch(-1)
)

val_ds = (
    val_dataset
    .batch(32)
    #.prefetch(-1)
)


# =============================================================================
# Helping functions
# =============================================================================

def multi_mcc_loss(y_true, y_pred, false_pos_penal=1.0):
    # Reshape image to flatten form
    y_true = K.reshape(y_true, [-1, y_true.shape[-1]])
    y_pred = K.reshape(y_pred, [-1, y_pred.shape[-1]])

    confusion_m = tf.matmul(K.transpose(y_true), y_pred)
    if false_pos_penal != 1.0:
      """
      This part is done for penalization of FalsePos symmetrically with FalseNeg,
      i.e. FalseNeg is favorized for the same factor. In such way MCC values are comparable.
      If you want to penalize FalseNeg, than just set false_pos_penal < 1.0 ;)
      """
      confusion_m = tf.linalg.band_part(confusion_m, 0, 0) + tf.linalg.band_part(confusion_m, 0, -1)*false_pos_penal + tf.linalg.band_part(confusion_m, -1, 0)/false_pos_penal
    
    N = K.sum(confusion_m)
    
    up = N*tf.linalg.trace(confusion_m) - K.sum(tf.matmul(confusion_m, confusion_m))
    down_left = K.sqrt(N**2 - K.sum(tf.matmul(confusion_m, K.transpose(confusion_m))))
    down_right = K.sqrt(N**2 - K.sum(tf.matmul(K.transpose(confusion_m), confusion_m)))
    
    mcc = up / (down_left * down_right + K.epsilon())
    mcc = tf.where(tf.math.is_nan(mcc), tf.zeros_like(mcc), mcc)
    
    return 1 - K.mean(mcc)

# =============================================================================
# Set up for simulation
# =============================================================================


all_history = {}


for navn, loss in [('CCE', 'categorical_crossentropy'), ('MCC', multi_mcc_loss)]:
    all_history[navn] = []
    for i in range(5):
        
        u = vanilla_unet()
        
        img1 = Input(shape=(128,128,399))
        f1 = F1Score(num_classes=6, average='micro')
        #K.clear_session()
        model = u.get_unet(img1, None, n_classes=6, last_activation='softmax')
        model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[f1])
        
        h = model.fit(train_ds,
              validation_data=(val_ds), 
              batch_size=32,
              epochs=1,
              verbose=2)
              #sample_weight=sample_weigths)
              
        all_history[navn].append(h)
 
        
print("lager json")
 
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(all_history, f, ensure_ascii=False, indent=4)







