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
from lib.add_xy_marks import add_xy_marks
xy = add_xy_marks()

from sklearn.decomposition import PCA


# =============================================================================
# Import data
# =============================================================================

img_70_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR70cm/2019_04_vnir70cm.hdr")
img_70_raw = spectral.SpyFile.load(img_70_raw)
img_30_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR30cm/2019_04_vnir30cm.hdr")
img_30_raw = spectral.SpyFile.load(img_30_raw)
nDSM_30 = Image.open("E:/M-DV-STeien/august2019/04/lidar/2019_04_nDSM_30cm_fitted.tif") 
nDSM_30 = np.array(nDSM_30)
roof_mask_30_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roof_mask_70_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_70cm.tif")
roof_mask_70_raw = np.array(roof_mask_70_raw)
roof_mask_30_raw = np.array(roof_mask_30_raw)

roads_mask_30_raw = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_veg_30cm.tif")
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

# =============================================================================
# Cut image in 1/4 for easier memory
# =============================================================================
hs_img = img_30_raw
roof_mask = combo_mask
nDSM = nDSM_30

cutoff = int(1169/2)
hs_img = hs_img[:,cutoff:cutoff*2,:] # NB!
roof_mask = roof_mask[:,cutoff:cutoff*2]
nDSM = nDSM[:,cutoff:cutoff*2]

# =============================================================================
# PCA on HS!
# =============================================================================
# df_hs = xy.transform_hs(np.array(hs_img), keep_all=True)
# pca = PCA(n_components=5)
# hs_pca = pd.DataFrame(pca.fit_transform(df_hs[[i for i in range(176)]]))
# hs_pca["x"] = df_hs.x
# hs_pca["y"] = df_hs.y

# hs_img_recreated = np.zeros((hs_img.shape[0], hs_img.shape[1], pca.n_components))

# for i, values in hs_pca.iterrows():
#     for dim in range(pca.n_components):
#         hs_img_recreated[int(values.x), int(values.y), dim] = values[dim]
        

# =============================================================================
# Add nDSM as a dimension
# =============================================================================
hs_pca_ndsm = np.dstack((hs_img, nDSM))


# =============================================================================
# Reduce to RGB  or 100 bands
# =============================================================================
img_rgb = np.dstack((hs_img[:,:,76], hs_img[:,:,46], hs_img[:,:,21]))
img_100bands = hs_img[:,:,:100]

# =============================================================================
# Combination of images
# =============================================================================
img_to_use = hs_pca_ndsm

X_shape = 128
y_shape = 128
n = 50

X_r,X_c,X_d = img_to_use.shape
y_r,y_c = roof_mask.shape


X = []
y = []

r_combinations = [i for i in np.linspace(0, X_r-X_shape, X_r-X_shape+1, dtype=int)]
c_combinations = [i for i in np.linspace(0, X_c-X_shape, X_c-X_shape+1, dtype=int)]

a,b = (np.random.choice(r_combinations, n), np.random.choice(c_combinations, n))
c,d = (a+X_shape, b+X_shape)

for a1,b1,c1,d1 in zip(a,b,c,d):
    X.append(img_to_use[a1:c1, b1:d1].reshape(X_shape,X_shape,X_d)) 
    y.append(roof_mask[a1:c1, b1:d1].reshape(y_shape,y_shape,1))
    
X = np.array(X)
y = np.array(y)

# =============================================================================
# Save data
# =============================================================================
np.save("E:/M-DV-STeien/august2019/04/temp_data/X_data_unseen.npy", X)
np.save("E:/M-DV-STeien/august2019/04/temp_data/y_data_unseen.npy", y)

