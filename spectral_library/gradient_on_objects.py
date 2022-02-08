# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 11:49:14 2022

@author: Stian
"""


from PIL import Image
import numpy as np
import pandas as pd
import spectral
import matplotlib.pyplot as plt

import spectral.io.envi as envi


# =============================================================================
# Load data
# =============================================================================

spec_lib = envi.open("E:/M-DV-STeien/juni2021/04/hs/2021_04_stack30cm_roof_speclib_python.hdr")
#spec_lib = spectral.SpyFile.load(spec_lib)
roofs = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roofs = np.array(roofs)

# =============================================================================
# Make df for spec_lib
# =============================================================================

df = pd.DataFrame(spec_lib.spectra, index=spec_lib.names, columns=spec_lib.bands.centers)

objects = df.index.unique()
df_objects = {}

for ting in objects:
    df_objects[ting] = {"snitt":np.gradient(df[df.index == ting].mean()),
                        "stdavvik":np.gradient(df[df.index == ting].std()) 
                       }

df_spectral_objects = pd.DataFrame(df_objects).T
df_spectral_objects = df_spectral_objects[df_spectral_objects.index == 'red concrete']

vnir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/VNIR30cm/2021_04_vnir30cm.hdr")
swir_raw = spectral.open_image("E:/M-DV-STeien/juni2021/04/hs/SWIR30cm/2021_04_swir30cm.hdr")

vnir = spectral.SpyFile.load(vnir_raw)
swir = spectral.SpyFile.load(swir_raw)

hs = np.dstack([vnir, swir])

# =============================================================================
# Find all the gradients
# =============================================================================


hs_small = hs[:, :1000, :]
gradients = []
for x in hs_small:
    for l in x:
        gradients.append(np.gradient(l))
        
gradients = np.array(gradients).reshape(hs_small.shape[0],
                                        hs_small.shape[1], 
                                        hs_small.shape[2])[:,:,:396]


x,y,d = gradients.shape
maps = []
for x_ in range(x):
    print(f"{x_} of {x}", end="\n")
    for y_ in range(y):
        if roofs[x_,y_] > .01:
            maps.append([np.square(gradients[x_,y_,:] \
                      - df_spectral_objects.iloc[i].snitt).mean() for i in range(10)])
        else:
            maps.append([0]*10)
            
maps = np.array(maps).reshape(hs_small.shape[0],
                              hs_small.shape[1], 
                              10)
arg = maps.argmin(axis=2)

x,y,d = maps.shape
maps2 = maps.copy()[:,:,0]
maps2[:,:] = -1
for x_ in range(x):
    for y_ in range(y):
        if maps[x_,y_,arg[x_,y_]] < 100:
            maps2[x_,y_] = arg[x_,y_]
            
            
plt.imshow(np.dstack([hs_small[:,:,76],hs_small[:,:,46],hs_small[:,:,21]])/2500)


maps2[maps2 > 600] = 0
plt.imshow(maps2)


            
