# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 13:36:05 2022

@author: Stian
"""

# Hent ut tak med det artikkelen bruker :)

import matplotlib.pyplot as plt
import numpy as np
from spectral import imshow, ndvi
from cv2 import Sobel, Laplacian, watershed
import cv2 as cv

import spectral
from PIL import Image


def NDVI(X):
    wave=ndvi(X,76,105)
    data_ndvi=np.nan_to_num(wave)
    return data_ndvi


nDSM = Image.open("E:/M-DV-STeien/august2019/04/lidar/2019_04_nDSM_30cm_fitted.tif") 
nDSM = np.array(nDSM)

roofs = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_bygning_30cm.tif")
roofs = np.array(roofs)

roads = Image.open("E:/M-DV-STeien/databaseFKB2019/04/04_veg_30cm.tif")
roads = np.array(roads)

vnir_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR30cm/2019_04_vnir30cm.hdr")
swir_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/SWIR30cm/2019_04_swir30cm.hdr")
vnir = spectral.SpyFile.load(vnir_raw)
swir = spectral.SpyFile.load(swir_raw)
hs = np.dstack([vnir, swir])

ndvi_mask = NDVI(hs)
ndvi_mask[ndvi_mask > 1] = 1
ndvi_mask[ndvi_mask < -1] = -1


plt.imshow(np.dstack([hs[:,:,76],hs[:,:,46],hs[:,:,21]])/100)

# Sobel
sobelx = cv.Sobel(nDSM,cv.CV_64F,1,0,ksize=3)
sobely = cv.Sobel(nDSM,cv.CV_64F,0,1,ksize=3)
sobelxy = np.sqrt(sobelx**2 + sobely**2)
plt.imshow(sobelxy)

# Høydeforskjellen mellom de to med mest forskjell

def slope(X):
    X = X[:,:]
    
    X = np.pad(X, ( # axis0
                      (1, 1), #axis1
                      (1, 1)), mode='constant', constant_values=100)
    
    maper = np.zeros((X.shape[0]-1, X.shape[1]-1))
    for x in range(1,X.shape[0]-1):
        for y in range(1,X.shape[1]-1):
            kernel = X[x-1:x+2, y-1:y+2]
            z = X[x,y]
            high = abs(kernel - z).max()
            maper[x,y] = high

            
    return maper

kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1,0,1]])


X = []
for i in range(nDSM.shape[1]):
    X.append(np.gradient(nDSM[:,i]))
X = np.array(X)
    
Y = []
for i in range(nDSM.shape[0]):
    Y.append(np.gradient(nDSM[i,:]))
Y = np.array(Y)



# Remove vegetation and everything below 1.6m.
hs[nDSM < 1.6] = 0
ndvi_mask = NDVI(hs)
ndvi_mask[ndvi_mask > 1] = 1
ndvi_mask[ndvi_mask < -1] = -1
hs[ndvi_mask > 0.3] = 0


# Remove road things
hs[roads > 0.01] = 0


# Filters
from scipy.ndimage import rotate
k = np.array([[1,1,1,1,1],
              [1,1,1,1,1],
              [1,1,1,1,1],
              [1,1,1,1,1],
              [1,1,1,1,1]])

k = np.pad(k, 1, mode='constant', constant_values=1)
filters = [k]

#k = k[1:-1, 1:-1]
#filters.append(k)

tri_down = np.tril(k)
tri_upper = np.triu(k)

flips = np.fliplr(tri_down)
flips_= np.fliplr(tri_upper)

filters.append(tri_down)
filters.append(tri_upper)
filters.append(flips)
filters.append(flips_)

#filters.append(np.pad(k, 1, mode='constant', constant_values=1))



img = np.uint8(hs[:,:,0])
img[img>0] = 255
img[img>0] = 1

kart = np.zeros((img.shape[0], img.shape[1]))
for k in filters:
    print(k)
    fasit = k.shape[0]*k.shape[1]
        #fasit = k.sum()
        
    x,y = k.shape
    for i in range(0, img.shape[0]-x):
        for j in range(0, img.shape[1]-y):
                
            chunk = img[i:i+x, j:j+y]
                
            if fasit == (chunk == k).sum():
                kart[i:i+x, j:j+y] = k




# Morph på den
img = np.uint8(hs[:,:,0])
img[img>0] = 255
kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations = 3)
erode = cv.erode(img, kernel, iterations = 1)
plt.imshow(opening)

hs[opening < 100] = 0
# Opening er masken


# Jac score FKB og mask
from sklearn.metrics import jaccard_score

roofs[roofs>0.001] = 1
roofs = roofs.astype(int)

jaccard_score(roofs.flatten(), kart.flatten())






