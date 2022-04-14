# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:50:31 2022

@author: Stian
"""

from PIL import Image
import spectral
import numpy as np
from lib.add_xy_marks import add_xy_marks
import matplotlib.pyplot as plt

xy = add_xy_marks()


# =============================================================================
# Load data
# =============================================================================

low_veg_mask = Image.open("E:/M-DV-STeien/august2019/04/2019_04_low_veg_mask_30cm.tif")
high_veg_mask = Image.open("E:/M-DV-STeien/august2019/04/2019_04_high_veg_mask_30cm.tif")

nDSM = Image.open("E:/M-DV-STeien/august2019/04/lidar/2019_04_nDSM_30cm_fitted.tif")
roof_mask = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")

hs_img = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR30cm/2019_04_vnir30cm.hdr")
hs_img = spectral.SpyFile.load(hs_img)


# =============================================================================
# Spectral char for low veg
# =============================================================================

mask = np.array(low_veg_mask)
mask_hs_img = hs_img.copy()
mask_hs_img[mask == 0] = 0

all_low_veg_points = xy.transform_hs(mask_hs_img)
low_mean = all_low_veg_points[[i for i in range(176)]].mean()
low_std = all_low_veg_points[[i for i in range(176)]].std()

low_veg_vis = np.dstack((mask_hs_img[:,:,76], mask_hs_img[:,:,46], mask_hs_img[:,:,21]))

centers = hs_img.bands.centers

plt.plot(centers, low_mean, label="mean low vegetation")
plt.fill_between(centers, low_mean - low_std, low_mean + low_std, alpha=0.2)
plt.xlabel("Wavelength $\mu$m"); plt.ylabel("Reflectance")
plt.grid()



# =============================================================================
# Spectral char for high veg
# =============================================================================

mask = np.array(high_veg_mask)
mask_hs_img = hs_img.copy()
mask_hs_img[mask == 0] = 0

all_high_veg_points = xy.transform_hs(mask_hs_img)
high_mean = all_high_veg_points[[i for i in range(176)]].mean()
high_std = all_high_veg_points[[i for i in range(176)]].std()

high_veg_vis = np.dstack((mask_hs_img[:,:,76], mask_hs_img[:,:,46], mask_hs_img[:,:,21]))

centers = hs_img.bands.centers

plt.plot(centers, high_mean, c="red", label="mean high vegetation")
plt.fill_between(centers, high_mean - high_std, high_mean + high_std, color="red",
                 alpha=0.2)
plt.xlabel("Wavelength $\mu$m"); plt.ylabel("Reflectance")
#plt.grid()



# =============================================================================
# Spectral char for roof tops
# =============================================================================

mask_nDSM = np.array(nDSM)

mask = np.array(roof_mask)
mask[mask>0.5] = 1 
mask_hs = hs_img.copy()
mask_hs[mask == 0] = 0

mask_hs[mask_nDSM < 1] = 0 

roof_vis = np.dstack((mask_hs[:,:,76], mask_hs[:,:,46], mask_hs[:,:,21]))

all_roof_points = xy.transform_hs(mask_hs)
roof_mean = all_roof_points[[i for i in range(176)]].mean()
roof_std = all_roof_points[[i for i in range(176)]].std()

centers = hs_img.bands.centers

plt.plot(centers, roof_mean, c="orange", label="mean roofs")
plt.fill_between(centers, roof_mean - roof_std, roof_mean + roof_std, color="orange",
                 alpha=0.2)
plt.xlabel("Wavelength $\mu$m"); plt.ylabel("Reflectance")
#plt.grid()
plt.legend(loc="upper left")
plt.show()


# =============================================================================
# Spectral char of points
# =============================================================================

import pandas as pd
data_points = pd.read_csv("E:/M-DV-STeien/august2019/04/sample_raster_points.csv")
#cols = ["navn"]
#cols.extend(['SAMPLE_'+str(i) for i in range(1,177)])

for row in data_points.iterrows():
    plt.plot(centers, row[1][['SAMPLE_'+str(i) for i in range(1,177)]],
             label=str(row[1].id) + " "  + str(row[1].navn))

plt.ylabel("Reflectance"); plt.xlabel("Wavelength $\mu$m")
plt.grid()
plt.legend(loc="upper left")
plt.show()
#plt.plot(data_points[cols])


