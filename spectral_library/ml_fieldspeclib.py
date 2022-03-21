# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 14:00:52 2022

@author: Stian
"""

# Find map based on field work


from PIL import Image
import numpy as np
import pandas as pd
import spectral
import matplotlib.pyplot as plt

import spectral.io.envi as envi

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from mycolorpy import colorlist as mcp

break

# =============================================================================
# Load data labels
# =============================================================================

spec_lib = envi.open("E:/M-DV-STeien/juni2021/04/hs/2021_04_roofs_classes.hdr")
area = spec_lib.asarray()
area = area.reshape(area.shape[0], area.shape[1])

roofs = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roofs = np.array(roofs)

# =============================================================================
# Load data HS 
# =============================================================================

vnir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/VNIR30cm/2021_04_vnir30cm.hdr")
swir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/SWIR30cm/2021_04_swir30cm.hdr")

vnir = spectral.SpyFile.load(vnir_raw)
swir = spectral.SpyFile.load(swir_raw)

hs = np.dstack([vnir, swir])

plt.imshow(np.dstack([hs[:,:,76],hs[:,:,46],hs[:,:,21]])/2500)

# =============================================================================
# Plot area image with nice colors
# =============================================================================

classes = ["None", "black concrete", "metal roofing", "black ceramic", "brown concrete", 
           "red concrete", "gravel", "green ceramic", "pcv", "tar roofing paper"]
colormap = ListedColormap(["black", "red", "green", "yellow", "cyan", "maroon",
                           "magenta", "seagreen", "purple", "blue"])
fig, ax = plt.subplots()
img = ax.imshow(area, cmap=colormap)
plt.show()


# =============================================================================
# Take out pixels of interests
# =============================================================================

hs[area==0]=0

X = hs[hs[:,:,0]!=0]
y = area[area!=0]

# =============================================================================
# Machine learning
# =============================================================================
lr = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier()
rf = RandomForestClassifier()

for estimator, name in [(lr, "lr"),
                         (svc, "svc"),
                         (knn, "knn"),
                         (rf, "rf")]:
    pl = make_pipeline(#StandardScaler(), 
                       PCA(n_components=10),
                       estimator)
    
    pl.fit(X,y)
    print(f"{name}: {pl.score(X,y)}")
# lr = LogisticRegression(C=1)
# lr.fit(X,
#         y)

# rf = RandomForestClassifier(verbose=1)
# rf.fit(X,
#         y)

# =============================================================================
# C for lr
# =============================================================================
C = [2,3,4,5,10,50,100]
for c in C:
    lr = LogisticRegression()
    pl = make_pipeline(#StandardScaler(), 
                       PCA(n_components=c),
                       lr)
    
    pl.fit(X,y)
    print(c, pl.score(X,y))


# =============================================================================
# Predict
# =============================================================================


pred = pl.predict(hs.reshape((hs.shape[0]*hs.shape[1],
                              hs.shape[2]))).reshape(hs.shape[0],hs.shape[1])

pred[roofs<0.01] = 0
plt.imshow(pred, cmap=colormap)


# =============================================================================
# Only use field work as pred
# =============================================================================

area = np.array(area)
area[roofs<0.01] = 10
area += 1
area[area==11] = 0
plt.imshow(area) 

np.save("roof_map.npy", area)



