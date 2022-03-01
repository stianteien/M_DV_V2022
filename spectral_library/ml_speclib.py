# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:42:17 2022

@author: Stian
"""

# Shallow ML on spectral lib


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

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from mycolorpy import colorlist as mcp


# =============================================================================
# Load data
# =============================================================================


spec_lib = envi.open("E:/M-DV-STeien/juni2021/04/hs/2021_04_stack30cm_roof_speclib_python.hdr")
#spec_lib = spectral.SpyFile.load(spec_lib)
roofs = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roofs = np.array(roofs)

df = pd.DataFrame(spec_lib.spectra, index=spec_lib.names, columns=spec_lib.bands.centers)
df.columns = np.floor(df.columns*1000)/1000

# =============================================================================
# Combine items with same strucutes, gray/red metal --> metal
# =============================================================================
# items_replace = {"black ceramic":"ceramic", 
#                   "black concrete":"concrete",
#                   "brown concrete":"concrete",
#                   "red concrete":"concrete",
#                   "dark metal": "metal",
#                   "grayish metal": "metal",
#                   "light metal": "metal",
#                   "red metal": "metal"}
# df = df.rename(index=items_replace)


df["label"] = df.index

df["label"].replace(df.index.unique().to_numpy(), 
                    [i for i in range(len(df.index.unique()))], inplace= True)

# =============================================================================
# Load data HS 
# =============================================================================

vnir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/VNIR30cm/2021_04_vnir30cm.hdr")
swir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/SWIR30cm/2021_04_swir30cm.hdr")



vnir = spectral.SpyFile.load(vnir_raw)
swir = spectral.SpyFile.load(swir_raw)

hs = np.dstack([vnir, swir])

hs_small = hs[:, :, :]
roofs = roofs[:, :]
plt.imshow(np.dstack([hs_small[:,:,76],hs_small[:,:,46],hs_small[:,:,21]])/2500)

# =============================================================================
# Shallow ML pixelwise
# =============================================================================
sc = StandardScaler()
sc.fit(df.drop(columns=["label"]))


knn = KNeighborsClassifier()
knn.fit(df.drop(columns=["label"]),
        df.label)

lr = LogisticRegression(max_iter=200)
lr.fit(df.drop(columns=["label"]),
        df.label)

svc = SVC(verbose=1)
svc.fit(df.drop(columns=["label"]),
        df.label)

rf = RandomForestClassifier(verbose=1)
rf.fit(df.drop(columns=["label"]),
        df.label)

# =============================================================================
# Get right wavelength for all sets
# =============================================================================

X = pd.DataFrame(hs_small.reshape(hs_small.shape[0]*hs_small.shape[1],hs_small.shape[2]),
                 columns=np.concatenate([vnir.bands.centers,swir.bands.centers]))

X.columns = np.floor(X.columns)/1000
#df.columns = np.concatenate(np.round(df.columns, 4), ["label"])
X = X.loc[:,~X.columns.duplicated()]

X = X[df.drop(columns=["label"]).columns]

# =============================================================================
# Predict a small set
# =============================================================================
def predict(estimator):
    roof = roofs.reshape(roofs.shape[0]*roofs.shape[1])
    test = X.to_numpy()[roof > 0.01]
    test_indx = np.argwhere(roof > 0.01)
    pred = estimator.predict(test)
    maps = np.zeros((hs_small.shape[0],
                     hs_small.shape[1])).reshape(hs_small.shape[0]*hs_small.shape[1]) -1
    maps[test_indx.flatten()] = pred
    maps = maps.reshape(hs_small.shape[0], hs_small.shape[1])
    pred = maps
    return pred

pred = predict(lr)

pred = knn.predict(X).reshape(hs_small.shape[0],hs_small.shape[1])
pred = lr.predict(X).reshape(hs_small.shape[0],hs_small.shape[1])
pred = svc.predict(X).reshape(hs_small.shape[0],hs_small.shape[1])
pred = rf.predict(X).reshape(hs_small.shape[0],hs_small.shape[1])

# =============================================================================
# Show results
# =============================================================================

pred[roofs < 0.01] = -1


ticks = ["None"]
ticks.extend(df.index.unique().to_list())

colors=mcp.gen_color(cmap="tab20",n=6)
colormap = ListedColormap(colors)

plt.imshow(t, cmap=colormap)
cbar = plt.colorbar(ticks=[0,1,2,3,4,5])
cbar.ax.set_yticklabels(new_classes.keys())
plt.show()


# =============================================================================
# Reduce class with earlier reuslts 
# =============================================================================

labels = np.load("label.npy")
t = ticks[labels]
    

classes=    {"None":"None",
     "eternit": "eternit",
     "tar roofing paper": "tar roofing paper",
     "black ceramic":"ceramic", 
                  "black concrete":"concrete",
                  "brown concrete":"concrete",
                  "red concrete":"concrete",
                  "dark metal": "metal",
                  "grayish metal": "metal",
                  "light metal": "metal",
                  "red metal": "metal"}

for i in classes:
    key = i
    value = classes[key]
    t[t == key] = value

new_classes = {"None": 0,
               "ceramic": 1,
               "concrete": 2,
               "eternit": 3,
               "metal": 4,
               "tar roofing paper":5}

for i in new_classes:
    key = i
    value = new_classes[key]
    t[t == key] = value

t.astype(int)

