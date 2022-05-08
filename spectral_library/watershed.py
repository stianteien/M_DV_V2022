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

img = np.load("roof_map.npy")
classes = ["None", "unknown", "black concrete", "metal roofing", "black ceramic", "brown concrete", 
           "red concrete", "gravel", "green ceramic", "pvc", "tar roofing paper"]

colormap = ListedColormap(["black", "gray", "red", "green", "yellow", "cyan", "maroon",
                           "magenta", "seagreen", "purple", "blue"])
fig, ax = plt.subplots()
ax.imshow(ss, cmap=colormap)
plt.show()

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

# img = np.uint8(img*23)

# markers, a = water(img, threshold=0.3) # start høyt, gå lavere

# a[markers == -1] = [255,0,0]
# plt.imshow(a)

# img[markers!=most] = 0; 
# =============================================================================
# Mark whole area as same roof
# =============================================================================

def fill_majority(markers, a):
    # Mye if else - se til senere om det kan fikses opp i.
    
    count = np.unique(markers, return_counts=True)
    most = count[0][count[1].argmax()]
    for i in range(1, np.unique(markers).max()+1):
        if i is not most:
            try:
                if np.bincount(a[markers==i][:,0]).shape[0] > 2:
                    # Then there is more than None and unknown in the object
                    most_off = np.bincount(a[markers==i][:,0]).argmax()
                    if most_off == 1:
                        # Pick out second most
                        l = np.argsort(np.bincount(a[markers==i][:,0]))[-2]
                        a[markers == i] = l
                    else:
                        a[markers == i] = most_off
                    
                else:
                    most_off = np.bincount(a[markers==i][:,0]).argmax()
                    a[markers == i] = most_off
            except:
                pass
    
    b = a.astype(int)
    b = b[:,:,0] -1 
    b[markers == -1] = -1
    
    return b, most

# =============================================================================
# Find most in a img
# =============================================================================
img = np.uint8(img)

markers, a = water(img, threshold=0.6)
b, most = fill_majority(markers, a)
ss = b
img[markers!=most] = 0

markers, a = water(img, threshold=0.5)
b, most = fill_majority(markers, a)
ss[b>-1] = b[b>-1]
img[markers!=most] = 0

markers, a = water(img, threshold=0.3)
b, most = fill_majority(markers, a)
ss[b>-1] = b[b>-1]
img[markers!=most] = 0


# =============================================================================
# Display results
# =============================================================================


plt.imshow(ss, cmap=colormap)
cbar = plt.colorbar(ticks=[-1,0,1,2,3,4,5,6,7,8,9], orientation='horizontal')
cbar.ax.set_xticklabels(classes, rotation=35, ha="right")
plt.show()

# =============================================================================
# Save map
# =============================================================================

np.save("label.npy", ss)

# =============================================================================
# Combine classes
# =============================================================================

ss += 1
ticks = np.array(ticks)
t = ticks[ss]
    

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

t = t.astype(int)


# Hvis den nye

colors=mcp.gen_color(cmap="tab10",n=np.unique(t).shape[0])
colormap = ListedColormap(colors)

plt.imshow(t, cmap=colormap)
cbar = plt.colorbar(ticks=[i for i in range(np.unique(t).shape[0])],
                    orientation='horizontal')
cbar.ax.set_xticklabels(new_classes, rotation=25, ha="right")
plt.show()







