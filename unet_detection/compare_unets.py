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
from models.serie_unet import serie_unet
from models.triple_serie_unet import triple_serie_unet
from models.unet_plus_plus import unet_plus_plus

# =============================================================================
# Set up env
# =============================================================================

seed = 123

vUnet = vanilla_unet()
unet2i = unet_2input()
unet2i_d = unet_2input_deep()
dUnet = double_unet()
tUnet = triple_unet()
sUnet = serie_unet()
tsUnet = triple_serie_unet()
unet_2plus = unet_plus_plus()

# =============================================================================
# Import data
# =============================================================================
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

def load_data(path="data/roofs/"):
    
    X_train = np.load(path+"X_data.npy") 
    X_70_train = np.load(path+"X_70_data.npy")
    y_train = np.load(path+"y_data.npy")
    
    X_val = np.load(path+"X_data_val.npy")
    X_70_val = np.load(path+"X_70_data_val.npy")
    y_val = np.load(path+"y_data_val.npy")
    
    X_test = np.load(path+"X_data_test.npy")
    X_70_test = np.load(path+"X_70_data_test.npy")
    y_test = np.load(path+"y_data_test.npy")

    
    
    y_train = redesign_y(y_train)
    y_val = redesign_y(y_val)
    y_test = redesign_y(y_test)

    # print(y_train.shape)
    # print(y_val.shape)
    # print(y_test.shape)
    
    # Scale image down so each border is the same size when upsizing!
    X_70_train = X_70_train[:,4:-4,4:-4,:]
    X_70_val = X_70_val[:,4:-4,4:-4,:]
    X_70_test = X_70_test[:,4:-4,4:-4,:]
    
    return X_train, X_70_train, y_train, \
           X_val, X_70_val, y_val, \
           X_test, X_70_test, y_test

# =============================================================================
# Set up for testing all nets
# =============================================================================

def run_sim(path="data/roofs/", save_name="f1_01"):

    X_train, X_70_train, y_train, X_val, X_70_val, y_val, X_test, X_70_test, y_test = load_data(path)

    f1_total = []
    names = []
    
    for net,name in [(vUnet, "vanilla unet"),
                (sUnet, "serie unet"),
                (unet_2plus, 'unet plus plus'),
                (unet2i_d, 'unet 2inputs depth'),
                (dUnet, "double unet")]:
            
        names.append(name)
        n = 10
        epochs = 150
        f1_scores = []
        jacards = []
        
        input_img1 = Input(shape=(128,128,X_train.shape[3]))
        input_img2 = Input(shape=(56,56,X_70_train.shape[3]))
        
        for i in range(n):
          # Build model
          model = None
          model = net.get_unet(input_img1, input_img2,
                                 n_classes=2, last_activation='softmax')
        
          model.compile(optimizer='adam',
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
          
          # Run net
          hh = model.fit([X_train, X_70_train],
                          y_train, 
                          batch_size=16,
                          epochs=epochs,
                          verbose=0)
          
          # Save scores
          pred = model.predict([X_test, X_70_test])
          f1 = f1_score(y_test.argmax(axis=3).flatten(), pred.argmax(axis=3).flatten())
          f1_scores.append(f1)
        
          print(f"{name}:\tRound {i+1} of {n}. F1 score: {f1}")
        
        
        f1_scores = np.array(f1_scores)
        f1_total.append(f1_scores)
    
    f1_total = np.array(f1_total)
        
    
    df = pd.DataFrame(f1_total.T, columns=names)
    df.to_csv(save_name+".csv", index=False)
  

# =============================================================================
# Run sim on all 3 datasets
# =============================================================================

for path, fname in [#('data/roofs/04/', "f1_04"),
                    #('data/roofs/05/', "f1_05"),
                    ('data/roofs/07/', "f1_07")]:
    
    run_sim(path, fname)