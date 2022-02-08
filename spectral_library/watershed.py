# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:30:34 2022

@author: Stian
"""

import numpy as np
import pandas as pd

import cv2 as cv


# =============================================================================
# Load data
# =============================================================================

img = np.load("roof_map.npy") + 1

# =============================================================================
# Watershed
# =============================================================================
# rgb_img = np.dstack([img,
#                        np.ones((img.shape[0],img.shape[1], 1)),
#                        np.ones((img.shape[0],img.shape[1], 1))])


# rgb_img = np.uint8(rgb_img*100)
# rgb = cv.cvtColor(rgb_img, cv.COLOR_RGB2GRAY)
# #img = cv.imread(img)

# #
# img = rgb
img = np.uint8(img*23)
ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY, 0)

kernel = np.ones((2,2),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)

sure_bg = cv.dilate(opening,kernel,iterations=3)

opening = opening.astype(np.uint8)

dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
ret, sure_fg = cv.threshold(dist_transform,0.2*dist_transform.max(),255,0)

sure_bg = np.uint8(sure_bg)
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

ret, markers = cv.connectedComponents(sure_fg)
markers = markers
markers[unknown==255] = 0

a = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

# Watershed
#gray = np.uint8(gray)
markers = cv.watershed(a,markers)

a[markers == -1] = [255,0,0]


