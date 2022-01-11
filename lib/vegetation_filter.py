# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:44:14 2021

@author: Stian
"""

import matplotlib.pyplot as plt
import numpy as np
from spectral import imshow, ndvi

class vegetation_filter:
    def __init__(self):
        pass
    
    
    def NDVI(self, X, threshold=0.6,
             red=76, NIR=105):
        """
        Return:
            X - filtered original image (only with vegetation)
            data_ndvi - ndvi mask for the image
        """
        wave=ndvi(X,red,NIR)
        data_ndvi=np.nan_to_num(wave)
        
        data_ndvi[data_ndvi[:,:,0] < 0.6] = 0
        X[data_ndvi[:,:,0] == 0] = 0
        return X, data_ndvi
    
    def lidar_heigth(self, img, lidar_img, threshold=0.3):
        """
        Return:
            Low height image - objects with lower than threshold
            High heiht image - objects with heigher than threshold
        """
        low_veg = img.copy()
        low_veg[lidar_img > threshold] = 0
        
        high_veg = img.copy()
        high_veg[lidar_img <= threshold] = 0
        
        return low_veg, high_veg