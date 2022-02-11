# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 12:49:50 2022

@author: Stian
"""

# Ml with watershed mask

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



# =============================================================================
# Load data
# =============================================================================

roofs = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roofs = np.array(roofs)

label = np.load("labels.npy")


vnir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/VNIR30cm/2021_04_vnir30cm.hdr")
swir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/SWIR30cm/2021_04_swir30cm.hdr")
vnir = spectral.SpyFile.load(vnir_raw)
swir = spectral.SpyFile.load(swir_raw)
hs = np.dstack([vnir, swir])


# =============================================================================
# Pick out area for learning
# =============================================================================

hs_train = hs[:,:500,:]
hs_test = hs[:,500:1000,:]
roof = roofs[:,500:1000]

train = hs_train[label != -1]

# =============================================================================
# LR train
# =============================================================================

lr = LogisticRegression()
lr.fit(train, label[label!=-1])


# =============================================================================
# LR test
# =============================================================================

pred = lr.predict(hs_test.reshape(hs_test.shape[0]*hs_test.shape[1],
                            hs_test.shape[2])).reshape(hs_test.shape[0],
                                                        hs_test.shape[1])

pred[roof < 0.05] = -1

# =============================================================================
# Display results
# =============================================================================


ticks = ["None"]
ticks.extend(['black concrete',
 'grayish metal',
 'light metal',
 'tar roofing paper',
 'eternit',
 'dark metal',
 'black ceramic',
 'red metal',
 'brown concrete',
 'red concrete'])


colors=mcp.gen_color(cmap="tab20",n=11)
colormap = ListedColormap(colors)

plt.imshow(pred, cmap=colormap)
cbar = plt.colorbar(ticks=[-1,0,1,2,3,4,5,6,7,8,9])
cbar.ax.set_yticklabels(ticks)
plt.show()

