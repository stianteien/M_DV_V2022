# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 16:11:24 2021

@author: Stian
"""

# Class making the x,y koords

import numpy as np
import pandas as pd

class add_xy_marks:
    def __init__(self):
        pass
    
    
    def transform_hs(self, data, keep_all=False):
        x_shape = data.shape[0]
        y_shape = data.shape[1]
        band_shape = data.shape[2]
        if keep_all:
            mask = zip(*np.where(data[:,:,1] > -1))
        else:
            mask = zip(*np.where(data[:,:,1])) # if value in (x,y,1) then find xy
        coords = [xy for xy in mask]
        temp_x = np.zeros((x_shape, y_shape))
        temp_y = np.zeros((x_shape, y_shape))
        for x,y in coords:
            temp_x[x,y] = x
            temp_y[x,y] = y 
        
        temp_x = temp_x.reshape(x_shape, y_shape, 1)
        temp_y = temp_y.reshape(x_shape, y_shape, 1)
        
        data = np.append(data, temp_x, axis=2) # Nest siste er x
        data = np.append(data, temp_y, axis=2) # Siste er y
        
        
        labels = [i for i in range(band_shape+2)]
        labels[-1] = "y"; labels[-2] = "x"
        image_flatten = data.reshape(x_shape*y_shape, band_shape + 2)
        df = pd.DataFrame(image_flatten, columns=labels)
        df = df.loc[~(df==0).all(axis=1)]

        #X = image_flatten[image_flatten[:,0] != 0]
        #X_coords = X[:, -2:]
        #X = X[:, :-2]
        
        return df
    
    def transform_1d(self, data):
        x_shape = data.shape[0]
        y_shape = data.shape[1]
        
        mask = zip(*np.where(data[:,:])) # if value in (x,y,1) then find xy
        coords = [xy for xy in mask]
        temp_x = np.zeros((x_shape, y_shape))
        temp_y = np.zeros((x_shape, y_shape))
        for x,y in coords:
            temp_x[x,y] = x
            temp_y[x,y] = y 
        
        temp_x = temp_x.reshape(x_shape, y_shape,1)
        temp_y = temp_y.reshape(x_shape, y_shape,1)
        data = data.reshape(x_shape, y_shape, 1)
        
        data = np.append(data, temp_x, axis=2) # Nest siste er x
        data = np.append(data, temp_y, axis=2) # Siste er y
        
        
        labels = [i for i in range(1+2)]
        labels[-1] = "y"; labels[-2] = "x"; labels[0] = "group"
        image_flatten = data.reshape(x_shape*y_shape, 1 + 2)
        df = pd.DataFrame(image_flatten, columns=labels)
        df = df.loc[~(df==0).all(axis=1)]
        
        return df
        