# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 13:01:47 2021

@author: Stian
"""

from PIL import Image
import numpy as np
import gdal

img = Image.open("E:/HS_DAT390_testarea_07august.tif")
img_array = np.array(img)

img_gdal = gdal.Open("E:/HS_DAT390_testarea_07august.tif")
bands = img_gdal.RasterCount
img_array = np.array(img_gdal.GetRasterBand(1).ReadAsArray())
for i in range(2,bands+1):
    print(f"band {i} of {bands}")
    a = np.array(img_gdal.GetRasterBand(i).ReadAsArray())
    img_array = np.dstack((img_array,a))

np.save("HS_testarea_07august.npy", img_array)