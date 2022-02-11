# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 15:30:34 2022

@author: Stian
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2 as cv

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from mycolorpy import colorlist as mcp


# =============================================================================
# Load data
# =============================================================================

img = np.load("roof_map.npy") + 1

# =============================================================================
# Watershed
# =============================================================================


def water(img, threshold=0.7):
    ret, thresh = cv.threshold(img,0,255,cv.THRESH_BINARY, 0)
    
    kernel = np.ones((2,2),np.uint8)
    opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
    
    sure_bg = cv.dilate(opening,kernel,iterations=3)
    
    opening = opening.astype(np.uint8)
    
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 3)
    ret, sure_fg = cv.threshold(dist_transform,threshold*dist_transform.max(),255,0)
    
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
    return markers, a

img = np.uint8(img*23)

markers, a = water(img, threshold=0.3) # start høyt, gå lavere

a[markers == -1] = [255,0,0]
plt.imshow(a)

# img[markers!=most] = 0; 
# =============================================================================
# Mark whole area as same roof
# =============================================================================


# 30 er alt...
count = np.unique(markers, return_counts=True)
most = count[0][count[1].argmax()]
for i in range(1, np.unique(markers).max()+1):
    if i is not most:
        try:
            most_off = np.bincount(a[markers==i][:,0]).argmax()
            a[markers == i] = most_off
        except:
            pass


b = a[:,:,0] / 23 -1 
b[markers == -1] = -1
b = b.astype(int)

# =============================================================================
# Combine different watersheds
# =============================================================================
ss[b>-1] = b[b>-1]


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

np.unique(b)

colors=mcp.gen_color(cmap="tab20",n=11)
colormap = ListedColormap(colors)

plt.imshow(ss, cmap=colormap)
cbar = plt.colorbar(ticks=[-1,0,1,2,3,4,5,6,7,8,9])
cbar.ax.set_yticklabels(ticks)
plt.show()



