# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:12:17 2022

@author: Stian
"""

from PIL import Image
import numpy as np
import spectral
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


# =============================================================================
# Import data
# =============================================================================

img_70_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR70cm/2019_04_vnir70cm.hdr")
img_70_raw = spectral.SpyFile.load(img_70_raw)
img_30_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR30cm/2019_04_vnir30cm.hdr")
img_30_raw = spectral.SpyFile.load(img_30_raw)
roof_mask_30_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roof_mask_70_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_70cm.tif")
roof_mask_70_raw = np.array(roof_mask_70_raw)

roof_mask_70_raw[roof_mask_70_raw > 0.01] = 1
#img_70[img_70 > 200] = 200

# =============================================================================
# Cut image in 1/4 for easier memory
# =============================================================================
cutoff = int(1169/4)
img_70 = img_70_raw[:,:cutoff,:]
roof_mask_70 = roof_mask_70_raw[:, :cutoff]

# =============================================================================
# Reduce to RGB  or 100 bands
# =============================================================================
img_70_rgb = np.dstack((img_70[:,:,76], img_70[:,:,46], img_70[:,:,21]))
img_70_100bands = img_70[:,:,:100]

# =============================================================================
# Combination of images
# =============================================================================
img_to_us = img_70

X_shape = 64
y_shape = 64
n = 50

X_r,X_c,X_d = img_to_us.shape
y_r,y_c = roof_mask_70.shape


X = []
y = []

r_combinations = [i for i in np.linspace(0, X_r-X_shape, X_r-X_shape+1, dtype=int)]
c_combinations = [i for i in np.linspace(0, X_c-X_shape, X_c-X_shape+1, dtype=int)]

a,b = (np.random.choice(r_combinations, n), np.random.choice(c_combinations, n))
c,d = (a+X_shape, b+X_shape)

for a1,b1,c1,d1 in zip(a,b,c,d):
    X.append(img_to_us[a1:c1, b1:d1].reshape(X_shape,X_shape,X_d)) 
    y.append(roof_mask_70[a1:c1, b1:d1].reshape(y_shape,y_shape,1))
    
X = np.array(X)
y = np.array(y)

# =============================================================================
# Save data
# =============================================================================
np.save("X_data.npy", X/100)
np.save("y_data.npy", y)

