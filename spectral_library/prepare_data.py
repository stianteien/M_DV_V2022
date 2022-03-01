# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:47:33 2022

@author: Stian
"""

from PIL import Image
import numpy as np
import pandas as pd
import spectral
import matplotlib.pyplot as plt
import spectral.io.envi as envi

from sklearn.linear_model import LogisticRegression

from mycolorpy import colorlist as mcp
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import datetime



# =============================================================================
# Load data
# =============================================================================

roofs = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roofs = np.array(roofs)

label = np.load("label_few.npy")

nDSM_30 = Image.open("E:/M-DV-STeien/juni2021/04/lidar/2021_04_nDSM_30cm_fitted.tif") 
nDSM_30 = np.array(nDSM_30)


vnir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/VNIR30cm/2021_04_vnir30cm.hdr")
swir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/SWIR30cm/2021_04_swir30cm.hdr")
vnir = spectral.SpyFile.load(vnir_raw)
swir = spectral.SpyFile.load(swir_raw)
hs = np.dstack([vnir, swir])


class data_maker:
    def __init__(self):
        pass
    
    def make_train_val_test(self, dist=[6,2,2]):
        assert sum(dist) == 10, "sum of dist != 10 (100%)"
        
        
        # Make train data
        print(f"{datetime.datetime.now()} - Starts making train set")
        self.make_set(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data.npy",
                
                 fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data.npy",
                 cutoff1=0,
                 cutoff2=dist[0],
                 n=150)
        
        # Make val data
        print(f"{datetime.datetime.now()} - Starts making val set")
        self.make_set(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data_val.npy",
                 
                 fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data_val.npy",
                 cutoff1=dist[0],
                 cutoff2=dist[0]+dist[1],
                 n=20)
        
        # Make test data
        print(f"{datetime.datetime.now()} - Starts making test set")
        self.make_set(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data_test.npy",
                 
                 fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data_test.npy",
                 cutoff1=dist[0]+dist[1],
                 cutoff2=dist[0]+dist[1]+dist[2],
                 n=50)
        
        
    # =============================================================================
    # Cut image for easier memory
    # =============================================================================
    def make_set(self, fnameX, fnamey, cutoff1, cutoff2, n):
        # Border big = 112 - border (8) på hver side
        # Border small = 112 / (7/3) = 48 , border til 64 (8 på hver side)
        
        hs_img = hs
        roof_mask = label
        
        cutoff1 = int(cutoff1/10 * hs_img.shape[1])
        cutoff2 = int(cutoff2/10 * hs_img.shape[1])
        
        
        hs_img = hs_img[:,cutoff1:cutoff2,:]
        roof_mask = roof_mask[:,cutoff1:cutoff2]
        nDSM = nDSM_30[:,cutoff1:cutoff2]
        

        # =============================================================================
        # Add nDSM as a dimension
        # =============================================================================
        hs_img_recreated = hs_img
        hs_pca_ndsm = np.dstack((hs_img_recreated, nDSM))
        
        
        # =============================================================================
        # Reduce to RGB  or 100 bands
        # =============================================================================
        #img_rgb = np.dstack((hs_img[:,:,76], hs_img[:,:,46], hs_img[:,:,21]))
        #img_100bands = hs_img[:,:,:100]
        
        # =============================================================================
        # Combination of images
        # =============================================================================
        img_to_use = hs_pca_ndsm
        
        X_shape = 112
        y_shape = 112

        X_r,X_c,X_d = img_to_use.shape
        y_r,y_c = roof_mask.shape
        
        
        X = []
        X_70 = []
        f = (7/3)
        y = []
        
        r_combinations = [i for i in np.linspace(0, X_r-X_shape, X_r-X_shape+1, dtype=int) ]
        c_combinations = [i for i in np.linspace(0, X_c-X_shape, X_c-X_shape+1, dtype=int) ]
        
        
        a,b = (np.random.choice(r_combinations, n), np.random.choice(c_combinations, n))
        c,d = (a+X_shape, b+X_shape)
        
        
        for a1,b1,c1,d1 in zip(a,b,c,d):
            X.append(img_to_use[a1:c1, b1:d1].reshape(X_shape,X_shape,X_d)) 
            y.append(roof_mask[a1:c1, b1:d1].reshape(y_shape,y_shape,1))
            
        X = np.array(X)
        X_70 = np.array(X_70)
        y = np.array(y)
        
        # =============================================================================
        # Make black border on all images
        # =============================================================================
        border_size = 8
        
        X = np.pad(X, ((0,0), # axis0
                      (border_size, border_size), #axis1
                      (border_size, border_size),
                      (0,0)), mode='constant', constant_values=0)
        
        
        y = np.pad(y, ((0,0), # axis0
                      (border_size, border_size), #axis1
                      (border_size, border_size),
                      (0,0)), mode='constant', constant_values=0)
    
        
        
        # =============================================================================
        # Save data
        # =============================================================================
        np.save(fnameX, X)
        np.save(fnamey, y)


# =============================================================================
# Acitivate function for files
# =============================================================================
d = data_maker()
d.make_train_val_test()