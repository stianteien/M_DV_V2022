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

label = np.load("label.npy")+1

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
    
    def make_train_val_test(self, dist=[6,2,2], train_chronological=False, test_chronological=False):
        assert sum(dist) == 10, "sum of dist != 10 (100%)"
        
        cutoffs = [(0, 4),
                   (5, 9),
                   (4,5),
                   (9,10)]
        
        train1_cut, train2_cut = cutoffs[0], cutoffs[1]
        #val1_cut, val2_cut = cutoffs[2], cutoffs[3]
        test1_cut, test2_cut = cutoffs[2], cutoffs[3]
        
        # Make train data
        if not train_chronological:
            print(f"{datetime.datetime.now()} - Starts making train set")
            self.make_set(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data.npy",
                     fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data.npy",
                     cutoffs=(train1_cut, train2_cut),
                     n=160)
        else:
            self.make_set_chronological(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data.npy",
                 fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data.npy",
                 cutoffs=(train1_cut, train2_cut))
        
        # Make val data
        # print(f"{datetime.datetime.now()} - Starts making val set")
        # self.make_set(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data_val.npy",
                 
        #          fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data_val.npy",
        #          cutoffs=(val1_cut, val2_cut),
        #          n=40)
        
        # Make test data
        if not test_chronological:
            print(f"{datetime.datetime.now()} - Starts making test set")
            self.make_set(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data_test.npy",        
                     fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data_test.npy",
                     cutoffs=(test1_cut, test2_cut),
                     n=40)
        else:
            self.make_set_chronological(fnameX="E:/M-DV-STeien/juni2021/04/temp_data/X_data_test.npy",
                 fnamey="E:/M-DV-STeien/juni2021/04/temp_data/y_data_test.npy",
                 cutoffs=(test1_cut, test2_cut))
        
        
    # =============================================================================
    # Cut image for easier memory
    # =============================================================================
    def make_set(self, fnameX, fnamey, cutoffs, n):
        # Border big = 112 - border (8) på hver side
        # Border small = 112 / (7/3) = 48 , border til 64 (8 på hver side)

        X = []
        y = []
        
        for c1, c2 in cutoffs:

            cutoff1 = int(c1/10 * hs.shape[1])
            cutoff2 = int(c2/10 * hs.shape[1])
            print(cutoff1, cutoff2, c1,c2, hs.shape[1])
        
            hs_img = hs[:,cutoff1:cutoff2,:]
            roof_mask = label[:,cutoff1:cutoff2]
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
            
            X_shape = 128
            y_shape = 128
    
            X_r,X_c,X_d = img_to_use.shape
            y_r,y_c = roof_mask.shape
            
            r_combinations = [i for i in np.linspace(0, X_r-X_shape, X_r-X_shape+1, dtype=int) ]
            c_combinations = [i for i in np.linspace(0, X_c-X_shape, X_c-X_shape+1, dtype=int) ]
        
        
            a,b = (np.random.choice(r_combinations, n), np.random.choice(c_combinations, int(n/2)))
            c,d = (a+X_shape, b+X_shape)
            
            print(roof_mask.shape)
            for a1,b1,c1,d1 in zip(a,b,c,d):
                X.append(img_to_use[a1:c1, b1:d1].reshape(X_shape,X_shape,X_d)) 
                y.append(roof_mask[a1:c1, b1:d1].reshape(y_shape,y_shape,1))
            
        X = np.array(X)
        y = np.array(y)
        
        # =============================================================================
        # Make black border on all images
        # =============================================================================
        # border_size = 8
        
        # X = np.pad(X, ((0,0), # axis0
        #               (border_size, border_size), #axis1
        #               (border_size, border_size),
        #               (0,0)), mode='constant', constant_values=0)
        
        
        # y = np.pad(y, ((0,0), # axis0
        #               (border_size, border_size), #axis1
        #               (border_size, border_size),
        #               (0,0)), mode='constant', constant_values=0)
    
        
        
        # =============================================================================
        # Save data
        # =============================================================================
        print(np.unique(y, return_counts=True))
        np.save(fnameX, X)
        np.save(fnamey, y)
        
        
    def make_set_chronological(self, fnameX, fnamey, cutoffs):

        X = []
        y = []
        
        
        for cutoff1, cutoff2 in cutoffs:
        
            cutoff1 = int(cutoff1/10 * hs.shape[1])
            cutoff2 = int(cutoff2/10 * hs.shape[1])
            
            
            hs_img = hs[:,cutoff1:cutoff2,:]
            roof_mask = label[:,cutoff1:cutoff2]
            nDSM = nDSM_30[:,cutoff1:cutoff2]
            
            hs_img = np.dstack((hs_img, nDSM))
            
            # =============================================================================
            # Find shape to cut into
            # =============================================================================
            X_shape = 128
            y_shape = 128
            
            nx,ny = np.floor(np.array(roof_mask.shape) / X_shape)
        
            
            for x_ in range(int(nx)):
                for y_ in range(int(ny)):
                    X.append(hs_img[int(x_*X_shape):int(x_*X_shape)+X_shape,
                                    int(y_*y_shape):int(y_*y_shape)+y_shape,:])
                    y.append(roof_mask[int(x_*X_shape):int(x_*X_shape)+X_shape,
                                       int(y_*y_shape):int(y_*y_shape)+y_shape])
                    
            # [1 2 3 4
            #  5 6 7 8]  <- Slik blir bildene
                
        X = np.array(X)
        y = np.array(y)
        
        print(np.unique(y, return_counts=True))
        print(f"nx:{nx}, ny:{ny}")
        np.save(fnameX, X)
        np.save(fnamey, y)
        
        
        # w = []
        # o = 0
        # for i in range(int(nx)):
        #     b = np.array(y[o])
        #     o += 1
        #     for j in range(1,int(ny)):
        #         b = np.append(b, y[o], axis=1)
        #         o += 1
                
        #     if len(w) == 0:
        #         w = b
        #     else:
        #         w = np.append(w, b, axis=0)
           
            


# =============================================================================
# Acitivate function for files
# =============================================================================
d = data_maker()
d.make_train_val_test(train_chronological=True, test_chronological=True)