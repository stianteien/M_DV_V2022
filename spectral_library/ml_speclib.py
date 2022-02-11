# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:42:17 2022

@author: Stian
"""

# Knn on spectral lib


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
df["label"] = df.index

df["label"].replace(df.index.unique().to_numpy(), 
                    [i for i in range(10)], inplace= True)

vnir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/VNIR30cm/2021_04_vnir30cm.hdr")
swir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/SWIR30cm/2021_04_swir30cm.hdr")



vnir = spectral.SpyFile.load(vnir_raw)
swir = spectral.SpyFile.load(swir_raw)

hs = np.dstack([vnir, swir])

hs_small = hs[:, :, :]
roofs = roofs[:, :]
plt.imshow(np.dstack([hs_small[:,:,76],hs_small[:,:,46],hs_small[:,:,21]])/2500)

#hs_small[roofs < 0.01] = 0


# Gradient forsøk
#gra = [np.gradient(v) for i,v in df.drop(columns=["label"]).iterrows()]
#df_g = pd.DataFrame(gra, 
#                    columns=df.drop(columns=["label"]).columns)


# =============================================================================
# Shallow ML pixelwise
# =============================================================================
sc = StandardScaler()
sc.fit(df.drop(columns=["label"]))


knn = KNeighborsClassifier()
knn.fit(df.drop(columns=["label"]),
        df.label)

lr = LogisticRegression(max_iter=100)
lr.fit(df.drop(columns=["label"]),
        df.label)

svc = SVC()
svc.fit(df.drop(columns=["label"]),
        df.label)

rf = RandomForestClassifier()
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

colors=mcp.gen_color(cmap="tab20",n=11)
colormap = ListedColormap(colors)

plt.imshow(pred, cmap=colormap)
cbar = plt.colorbar(ticks=[-1,0,1,2,3,4,5,6,7,8,9])
cbar.ax.set_yticklabels(ticks)
plt.show()




