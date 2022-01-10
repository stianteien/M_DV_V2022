# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:23:10 2022

@author: Stian
"""


from PIL import Image
import numpy as np
import spectral

import matplotlib.pyplot as plt

from lib.vegetation_filter import vegetation_filter

filters = vegetation_filter()


# =============================================================================
# Load data
# =============================================================================

hs_img = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR30cm/2019_04_vnir30cm.hdr")
nDSM_img = Image.open("E:/M-DV-STeien/august2019/04/lidar/2019_04_nDSM_30cm_fitted.tif") 

hs_img = spectral.SpyFile.load(hs_img)

# =============================================================================
# Filter on NDVI and nDSM
# =============================================================================

veg_img,_ = filters.NDVI(hs_img)

low_veg, high_veg = filters.lidar_heigth(hs_img, 
                                         np.array(nDSM_img),
                                         threshold=1)

# =============================================================================
# Make mask and save
# =============================================================================

low_veg_mask = low_veg[:,:,0]
low_veg_mask[low_veg_mask > 0] = 1

high_veg_mask = high_veg[:,:,0]
high_veg_mask[high_veg_mask > 0] = 1

im = Image.fromarray(low_veg_mask)
im.save("E:/M-DV-STeien/august2019/04/lidar/2019_04_low_veg_mask_30cm.tif",
        tiffinfo=nDSM_img.tag, save_all=True)

im = Image.fromarray(high_veg_mask)
im.save("E:/M-DV-STeien/august2019/04/lidar/2019_04_high_veg_mask_30cm.tif",
        tiffinfo=nDSM_img.tag, save_all=True)

