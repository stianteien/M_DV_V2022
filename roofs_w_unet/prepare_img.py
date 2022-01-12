# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 14:12:17 2022

@author: Stian
"""

from PIL import Image
import numpy as np
import spectral
import matplotlib.pyplot as plt


# =============================================================================
# Import data
# =============================================================================

img_70 = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR70cm/2019_04_vnir70cm.hdr")
img_70 = spectral.SpyFile.load(img_70)
img_30 = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR30cm/2019_04_vnir30cm.hdr")
img_30 = spectral.SpyFile.load(img_30)
roof_mask_30 = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roof_mask_70 = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_70cm.tif")
roof_mask_70 = np.array(roof_mask_70)

roof_mask_70[roof_mask_70 > 0.01] = 1
#img_70[img_70 > 200] = 200

# =============================================================================
# Cut image in 1/4 for easier memory
# =============================================================================
img_70 = img_70[:,:int(1169/4),:]
roof_mask_70 = roof_mask_70[:, :int(1169/4)]

# =============================================================================
# Reduce to RGB 
# =============================================================================
img_70_rgb = np.dstack((img_70[:,:,76], img_70[:,:,46], img_70[:,:,21]))


# =============================================================================
# Combination of images
# =============================================================================

X_shape = 64
y_shape = 64
n = 50

X_r,X_c,X_d = img_70_rgb.shape
y_r,y_c = roof_mask_70.shape


X = []
y = []

r_combinations = [i for i in np.linspace(0, X_r-X_shape, X_r-X_shape+1, dtype=int)]
c_combinations = [i for i in np.linspace(0, X_c-X_shape, X_c-X_shape+1, dtype=int)]

#a,b = (np.random.choice(r_combinations, n), np.random.choice(c_combinations, n))

for _ in range(n):
    a,b = (np.random.choice(r_combinations), np.random.choice(c_combinations))
    c,d = (a,b)
 
    X.append(img_70_rgb[a:a+X_shape, b:b+X_shape].reshape(X_shape,X_shape,X_d))  
    y.append(roof_mask_70[c:c+y_shape, d:d+y_shape].reshape(y_shape,y_shape,1))

X = np.array(X)
y = np.array(y)

# =============================================================================
# Save data
# =============================================================================
np.save("X_data.npy", X/100)
np.save("y_data.npy", y/100)

