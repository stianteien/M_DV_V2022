# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 16:43:11 2022

@author: Stian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import f1_score

from tensorflow.keras.layers import Input 
#from keras.utils.vis_utils import plot_model

from models.vanilla_unet import vanilla_unet
from models.unet_2input import unet_2input
from models.unet_2input_deep import unet_2input_deep
from models.double_unet import double_unet
from models.triple_unet import triple_unet

# =============================================================================
# Set up env
# =============================================================================

seed = 123

vUnet = vanilla_unet()
unet2i = unet_2input()
unet2i_d = unet_2input_deep()
dUnet = double_unet()
tUnet = triple_unet()

# =============================================================================
# Import data
# =============================================================================

X_train = np.load("data/roofs/X_data.npy") 
X_70_train = np.load("data/roofs/X_70_data.npy")
y_train = np.load("data/roofs/y_data.npy")

X_val = np.load("data/roofs/X_data_val.npy")
X_70_val = np.load("data/roofs/X_70_data_val.npy")
y_val = np.load("data/roofs/y_data_val.npy")

X_test = np.load("data/roofs/X_data_test.npy")
X_70_test = np.load("data/roofs/X_70_data_test.npy")
y_test = np.load("data/roofs/y_data_test.npy")

def redesign_y(y):
  n,r1,c1,d = y.shape
  # Adds a new dimension of layer too have two class problem.
  yy = np.append(y, np.zeros((n, r1, c1,d)), axis=3)
  for i in range(int(y.max()-1)):  
    yy = np.append(yy, np.zeros((n, r1, c1,d)), axis=3)
  yy1 = yy.copy()
  yy1[:,:,:,0] = 0 # reset map
  for i in range(n):
    values = yy[i,:,:,0]
    for r in range(r1):
      for c in range(c1):
        value = yy[i,r,c,0]
        yy1[i,r,c,int(value)] = 1

  return yy1

y_train = redesign_y(y_train)
y_val = redesign_y(y_val)
y_test = redesign_y(y_test)

# print(y_train.shape)
# print(y_val.shape)
# print(y_test.shape)
x70u = X_70_train.copy()
x70u_val = X_70_val.copy()
x70u_test = X_70_test.copy()

# Scale image down so each border is the same size when upsizing!
X_70_train = X_70_train[:,4:-4,4:-4,:]
X_70_val = X_70_val[:,4:-4,4:-4,:]
X_70_test = X_70_test[:,4:-4,4:-4,:]

# =============================================================================
# Set up for testing all nets
# =============================================================================
f1_total = []

for net,name in [
            (vUnet, "vanilla unet")]:
        
    n = 10
    epochs = 150
    f1_scores = []
    jacards = []
    
    input_img1 = Input(shape=(128,128,177))
    
    for i in range(n):
      # Build model
      model = None
      model = net.get_unet(input_img1,
                             n_classes=2, last_activation='softmax')
    
      model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
      
      # Run net
      hh = model.fit([X_train],
                      y_train, 
                      batch_size=16,
                      epochs=epochs,
                      verbose=0)
      
      # Save scores
      pred = model.predict([X_test])
      f1 = f1_score(y_test.argmax(axis=3).flatten(), pred.argmax(axis=3).flatten())
      f1_scores.append(f1)
    
      print(f"{name}:\tRound {i+1} of {n}. F1 score: {f1}")
    
    
    f1_scores = np.array(f1_scores)
    f1_total.append(f1_scores)



for net,name in [
            (unet2i, "doubleinput unet"),
            (unet2i_d, "doubleinput deep unet"),
            (dUnet, "double unet"),
            (tUnet, "triple unet")]:
        
    n = 10
    epochs = 150
    f1_scores = []
    jacards = []
    
    input_img1 = Input(shape=(128,128,177))
    input_img2 = Input(shape=(56,56,220))
    
    for i in range(n):
      # Build model
      model = None
      model = net.get_unet(input_img1, input_img1,
                             n_classes=2, last_activation='softmax')
    
      model.compile(optimizer='adam',
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
      
      # Run net
      hh = model.fit([X_train],
                      y_train, 
                      batch_size=16,
                      epochs=epochs,
                      verbose=0)
      
      # Save scores
      pred = model.predict([X_test])
      f1 = f1_score(y_test.argmax(axis=3).flatten(), pred.argmax(axis=3).flatten())
      f1_scores.append(f1)
    
      print(f"{name}:\tRound {i+1} of {n}. F1 score: {f1}")
    
    
    f1_scores = np.array(f1_scores)
    f1_total.append(f1_scores)

f1_total = np.array(f1_total)
    

df = pd.DataFrame(f1_total.T, columns=["vanilla unet", "doubleinput unet",
                                           "doubleinput deep unet","double unet",
                                           "triple unet"])
df.to_csv("f1_total_test.csv", index=False)
  