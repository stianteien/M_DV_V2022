# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:01:39 2022

@author: Stian
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import datetime

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mycolorpy import colorlist as mcp
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping

from M_DV_V2022.unet_detection.models.vanilla_unet import vanilla_unet

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.models import Model

import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras import backend as K
from segmentation_models.losses import CategoricalFocalLoss

import itertools

# ============
# Load data
# ============

X_train_raw = np.load("data/roofs/X_data.npy") 
y_train_raw = np.load("data/roofs/y_data.npy")

X_test_raw = np.load("data/roofs/X_data_test.npy")
y_test_raw = np.load("data/roofs/y_data_test.npy")


# ============
# Transform, reshape and split data
# ============


def redesign_y(y):
  y = y.reshape((y.shape[0],y.shape[1], y.shape[2], 1))
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


def train_val_split(X, y, val_split=0.2, idx=None):
  # Check if possible to stride
  assert np.unique(y).shape[0] == 10, "not enough unique values in set to stride" # n classes / Kanskje ta med det?

  # Do it
  dist = []
  for y_ in y:
    a = list(np.unique(y_, return_counts=True))
    for i in range(11):
      if i not in a[0]:
        a[0] = np.append(a[0], i)
        a[1] = np.append(a[1], 0)

    a[0], a[1] = zip(*sorted(zip(a[0], a[1])))
    dist.append(a)

  dist = np.array(dist, dtype=object)
  data_length = np.array([i for i in range(X.shape[0])])
  val_split = int(X.shape[0]*val_split)

  switch_test = False
  if idx is None:
    for _ in range(100):
      idx = np.random.choice(data_length, replace=False, size=val_split)
      e = np.sum(dist[idx], axis=0)[1]
      test = np.any((e == 0))
      if not test:
        print(test, e, idx)
        # ok in test set?
        switch_test = True
        break
    
    assert switch_test == True, "Not found any good strides"
  X_val = X[idx]
  y_val = y[idx]

  not_idx = np.array(data_length[list(set(range(X.shape[0])) - set(idx))])
  X_train = X[not_idx]
  y_train = y[not_idx]

  a = np.unique(y_val ,return_counts=True)[1]
  b = np.unique(y_train ,return_counts=True)[1]
  try: 
    c = np.abs(a/np.sum(a) - b/np.sum(b))
  except:
    c = np.array([1,1])
  

  t = (c < 0.01).all()
  print(c)
  

  return X_train, X_val, y_train, y_val, t


X_train, X_val, y_train, y_val, t = train_val_split(X_train_raw, y_train_raw, val_split=0.2,
                                                    idx=np.array([94, 26, 69, 34 ,75, 83, 5, 40 , 7, 60 , 8, 20, 92, 55, 78, 85, 63, 65 ,80]))




y_train = redesign_y(y_train)
y_val = redesign_y(y_val)
y_test = redesign_y(y_test_raw)

# ============
# Scale the data!
# ============
X_test = X_test_raw.copy()

maksen = 32767
max_height = 30
X_train[:,:,:,:-1][X_train[:,:,:,:-1]>maksen] = maksen
X_val[:,:,:,:-1][X_val[:,:,:,:-1]>maksen] = maksen
X_test[:,:,:,:-1][X_test[:,:,:,:-1]>maksen] = maksen

X_train[:,:,:,:-1] /= maksen
X_val[:,:,:,:-1] /= maksen
X_test[:,:,:,:-1] /= maksen

X_train[:,:,:,-1:] /= max_height
X_val[:,:,:,-1:] /= max_height
X_test[:,:,:,-1:] /= max_height

X_train = abs(X_train)
X_val = abs(X_val)
X_test = abs(X_test)

n = y_train.shape[-1]

print(n)

# ============
# Set up mcc
# ============

def multi_mcc(y_true, y_pred, false_pos_penal=1.0):
    # Reshape image to flatten form
    y_true = K.reshape(y_true, [-1, n]) # classes
    y_pred = K.reshape(y_pred, [-1, n])

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
    return K.mean(mcc)


# ============
# Set up model
# ============

import time

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)



from segmentation_models.losses import CategoricalFocalLoss
from segmentation_models.losses import JaccardLoss
from segmentation_models.losses import DiceLoss
from segmentation_models.losses import CategoricalCELoss

focal_loss = CategoricalFocalLoss()
jac_loss = JaccardLoss()
dice_loss = DiceLoss()
cce_loss = CategoricalCELoss()
focal_jac = jac_loss + focal_loss


loss_comb = [("Focal loss", CategoricalFocalLoss()),
             ("Jaccard loss", JaccardLoss()),
             ("Dice loss", DiceLoss()),
             ("CCE loss", CategoricalCELoss()),
             ("Focal + Jaccard", CategoricalFocalLoss() + JaccardLoss()),
             ("Focal + 0.5*Jaccard",  CategoricalFocalLoss() + 0.5*JaccardLoss() ),
             ("Focal + Dice", CategoricalFocalLoss() + DiceLoss())]

model_comb = [("Unet", "unet"),
              ("ResNet34", "resnet34"),
              ("ResNet50", "resnet50")]

opti_comb = [("Adam", 'adam'),
             ("SDG", 'sgd'),
             ("RMSprop", "rmsprop")]

comb = [loss_comb, model_comb, opti_comb]
my_comb = (list(itertools.product(*comb)))



def make_model(loss, name, op, size):
    if name != "unet":
        model = sm.Unet(backbone_name=name, encoder_weights=None,
                        input_shape=(128, 128, 399),
                        classes=n, activation='softmax')
    else:
        u = vanilla_unet()
        inp = Input(shape=(128, 128, 399))
        model = u.get_unet(inp, None, n_classes=10, last_activation="softmax")
        
    model.compile(optimizer=op,
                  loss=loss,
                  metrics=[multi_mcc])
    
    return model
    

def train_model(model, i, X, X_v):
    time_callback = TimeHistory()
    h = model.fit(X, y_train,
              validation_data=(X_v, y_val),
              callbacks=[time_callback],
              epochs=250,
              verbose=0)
    if i == 9:
        print(model.evaluate(X_val, y_val))
    #model.save_weights("model_"+str(i)+".h5")
    
    return h.history["val_multi_mcc"]


df = pd.DataFrame([])
for loss, net, op in my_comb:
    names = "\n".join([loss[0], net[0], op[0]])
    loss, net, op = loss[1], net[1], op[1]
    
    print(f"[{datetime.datetime.now()}] working with {names}")
    temp = []
    #l = JaccardLoss() + CategoricalFocalLoss()
    for i in range(10):
        #print(f"[{datetime.datetime.now()}] working with {i}")
        model = make_model(loss, net, op, X_train.shape[-1])
        val_mcc = train_model(model, i, X_train, X_val)
        
        df[names+"_val_mcc_"+str(i)] = val_mcc
       


    #temp = np.array(temp)
    #df[name] = np.mean(temp,axis=0)
    #df[name+"_std"] = np.std(temp, axis=0)
    
    

#df = pd.DataFrame(all_me)
    df.to_csv("final_massive_test.csv")






