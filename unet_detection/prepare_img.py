# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:12:17 2022

@author: Stian
"""

from PIL import Image
import numpy as np
import pandas as pd
import spectral
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
import datetime


# =============================================================================
# Import data
# =============================================================================

img_70_raw = spectral.open_image("E:/M-DV-STeien/august2019/07/hs/VNIR70cm/2019_07_vnir70cm.hdr")
img_70_raw = spectral.SpyFile.load(img_70_raw)
img_70_swir = spectral.open_image("E:/M-DV-STeien/august2019/07/hs/SWIR70cm/2019_07_swir70cm_tostack.hdr")
img_70_swir = spectral.SpyFile.load(img_70_swir)

img_30_raw = spectral.open_image("E:/M-DV-STeien/august2019/07/hs/VNIR30cm/2019_07_vnir30cm.hdr")
img_30_raw = spectral.SpyFile.load(img_30_raw)

nDSM_30 = Image.open("E:/M-DV-STeien/august2019/07/lidar/2019_07_nDSM_30cm_fitted.tif") 
nDSM_30 = np.array(nDSM_30)

roof_mask_30_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/07/07_bygning_30cm.tif")
roof_mask_70_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/07/07_bygning_70cm.tif")
roof_mask_70_raw = np.array(roof_mask_70_raw)
roof_mask_30_raw = np.array(roof_mask_30_raw)

roads_mask_30_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/07/07_veg_30cm.tif")
roads_mask_30_raw = np.array(roads_mask_30_raw)

roof_mask_70_raw[roof_mask_70_raw > 0.01] = 1
roof_mask_30_raw[roof_mask_30_raw > 0.01] = 1
roads_mask_30_raw[roads_mask_30_raw > 0.01] = 1
#img_70[img_70 > 200] = 200

# =============================================================================
# Combine roads and roofs
# none  - 0
# roofs - 1
# roads - 2
# =============================================================================
roads_mask = roads_mask_30_raw.copy()
roads_mask[roads_mask == 1] = 2
combo_mask = roads_mask.copy()
combo_mask += roof_mask_30_raw
combo_mask[combo_mask > 2] = 1


# =============================================================================
# Images to use
# =============================================================================


class data_maker:
    def __init__(self):
        pass
    
    def make_train_val_test(self, dist=[6,2,2]):
        assert sum(dist) == 10, "sum of dist != 10 (100%)"
        
        self.pca = PCA(n_components=5)
        self.pca_70 = PCA(n_components=5)
        
        # Make train data
        print(f"{datetime.datetime.now()} - Starts making train set")
        self.make_set(fnameX="E:/M-DV-STeien/august2019/07/temp_data/X_data.npy",
                 fnameX_70="E:/M-DV-STeien/august2019/07/temp_data/X_70_data.npy",
                 fnamey="E:/M-DV-STeien/august2019/07/temp_data/y_data.npy",
                 cutoff1=0,
                 cutoff2=dist[0],
                 n=1000)
        
        # Make val data
        print(f"{datetime.datetime.now()} - Starts making val set")
        self.make_set(fnameX="E:/M-DV-STeien/august2019/07/temp_data/X_data_val.npy",
                 fnameX_70="E:/M-DV-STeien/august2019/07/temp_data/X_70_data_val.npy",
                 fnamey="E:/M-DV-STeien/august2019/07/temp_data/y_data_val.npy",
                 cutoff1=dist[0],
                 cutoff2=dist[0]+dist[1],
                 n=200)
        
        # Make test data
        print(f"{datetime.datetime.now()} - Starts making test set")
        self.make_set(fnameX="E:/M-DV-STeien/august2019/07/temp_data/X_data_test.npy",
                 fnameX_70="E:/M-DV-STeien/august2019/07/temp_data/X_70_data_test.npy",
                 fnamey="E:/M-DV-STeien/august2019/07/temp_data/y_data_test.npy",
                 cutoff1=dist[0]+dist[1],
                 cutoff2=dist[0]+dist[1]+dist[2],
                 n=500)
        
        
    # =============================================================================
    # Cut image for easier memory
    # =============================================================================
    def make_set(self, fnameX, fnameX_70, fnamey, cutoff1, cutoff2, n):
        # Border big = 112 - border (8) p?? hver side
        # Border small = 112 / (7/3) = 48 , border til 64 (8 p?? hver side)
        
        hs_img = img_30_raw
        hs_img_70 = img_70_swir
        roof_mask = roof_mask_30_raw
        nDSM = nDSM_30
        
        cutoff1 = int(cutoff1/10 * hs_img.shape[1])
        cutoff2 = int(cutoff2/10 * hs_img.shape[1])
        
        
        hs_img = hs_img[:,cutoff1:cutoff2,:]
        hs_img_70 = hs_img_70[:,int(cutoff1/(7/3)):int(cutoff2/(7/3)),:]
        roof_mask = roof_mask[:,cutoff1:cutoff2]
        nDSM = nDSM[:,cutoff1:cutoff2]
        
        # =============================================================================
        # PCA on HS!
        # =============================================================================
        x,y,d = hs_img.shape
        x_70,y_70,d_70 = hs_img_70.shape
        if cutoff1 == 0:
            self.pca.fit(hs_img.reshape((x*y, d)))
            self.pca_70.fit(hs_img_70.reshape((x_70*y_70,d_70)))
        
        hs_img = self.pca.transform(hs_img.reshape((x*y, d))).reshape((x,y,self.pca.n_components))
        hs_img_70 = self.pca_70.transform(hs_img_70.reshape((x_70*y_70, d_70))).reshape((x_70,y_70,self.pca_70.n_components))
                
        
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
        
        _,_,X_70_d = hs_img_70.shape
        X_r,X_c,X_d = img_to_use.shape
        y_r,y_c = roof_mask.shape
        
        
        X = []
        X_70 = []
        f = (7/3)
        y = []
        
        r_combinations = [i for i in np.linspace(0, X_r-X_shape, X_r-X_shape+1, dtype=int) 
                    if (i/(7/3)).is_integer()]
        c_combinations = [i for i in np.linspace(0, X_c-X_shape, X_c-X_shape+1, dtype=int) 
                    if (i/(7/3)).is_integer()]
        
        
        a,b = (np.random.choice(r_combinations, n), np.random.choice(c_combinations, n))
        c,d = (a+X_shape, b+X_shape)
        
        
        for a1,b1,c1,d1 in zip(a,b,c,d):
            X_70.append(hs_img_70[int(round(a1/f)):int(round(c1/f)),
                                   int(round(b1/f)):int(round(d1/f))])
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
        
        X_70 = np.pad(X_70, ((0,0), # axis0
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
        np.save(fnameX_70, X_70)
        np.save(fnamey, y)


# =============================================================================
# Acitivate function for files
# =============================================================================
d = data_maker()
d.make_train_val_test()
