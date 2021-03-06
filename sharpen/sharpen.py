# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 13:09:19 2022

@author: Stian
"""

# =============================================================================
# Imports
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import spectral

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, \
                         UpSampling2D, Conv2DTranspose, MaxPooling2D



# =============================================================================
# Load data
# =============================================================================

#img_70 = np.load("E:/M-DV-STeien/Sharping_test/2019_04_vnir70cm_utdrag.npy")
#img_30 = np.load("E:/M-DV-STeien/Sharping_test/2019_04_vnir30cm_utdrag.npy")
img_70_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR70cm/2019_04_vnir70cm.hdr")
img_70_raw = spectral.SpyFile.load(img_70_raw)
img_30_raw = spectral.open_image("E:/M-DV-STeien/august2019/04/hs/VNIR30cm/2019_04_vnir30cm.hdr")
img_30_raw = spectral.SpyFile.load(img_30_raw)


# =============================================================================
# Reduce to fitting size and use only RGB
# =============================================================================
img_30_rgb = np.dstack((img_30_raw[:,:,76],
                        img_30_raw[:,:,46],
                        img_30_raw[:,:,21]))
img_70_rgb = np.dstack((img_70_raw[:,:,76],
                        img_70_raw[:,:,46],
                        img_70_raw[:,:,21]))

img_30_gray = img_30_raw[:,:,76]
img_70_gray = img_70_raw[:,:,76]


# =============================================================================
# Extract n amount of images from image
# =============================================================================

n = 50

# SHORTCUT DENNER EKKE GREI EGENTLIG!
img_70_gray = img_70_rgb
img_30_gray = img_30_rgb
img_70_gray[img_70_gray > 200] = 200
img_30_gray[img_30_gray > 200] = 200
img_70_gray = img_70_gray/200
img_30_gray = img_30_gray/200


X_r,X_c,X_d = img_70_gray.shape
y_r,y_c,y_d = img_30_gray.shape

X_shape = 55
y_shape = 128
    
    
X = []
y = []
    
r_combinations = [i for i in np.linspace(0, X_r-X_shape, X_r-X_shape+1, dtype=int)]
c_combinations = [i for i in np.linspace(0, X_c-X_shape, X_c-X_shape+1, dtype=int)]

 
a,b = (np.random.choice(r_combinations, n), np.random.choice(c_combinations, n))
c,d = (np.floor(a*(128/55)), np.floor(b*(128/55)))


for a1,b1,c1,d1 in zip(a,b,c,d):
    c1 = int(c1)
    d1 = int(d1)
    X.append(img_70_gray[a1:a1+X_shape, b1:b1+X_shape].reshape(X_shape,X_shape,X_d)) 
    y.append(img_30_gray[c1:c1+y_shape, d1:d1+y_shape].reshape(y_shape,y_shape,y_d))
        
X = np.array(X)
y = np.array(y)
    

np.save("X_data.npy", X)
np.save("y_data.npy", y)


# =============================================================================
# Make model  (in 3x3, out 7x7)
# =============================================================================

model = Sequential()

model.add(Input(shape=(60,60,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu',
                     padding='valid'))
model.add(MaxPooling2D(2))
model.add(Conv2D(32, kernel_size=3, activation='relu',
                padding='same'))
model.add(UpSampling2D(5))
model.add(Conv2D(32, kernel_size=3, activation='relu',
                 padding='valid'))
model.add(MaxPooling2D(2))
model.add(UpSampling2D(2))

model.add(Conv2D(1, kernel_size=3, activation='linear',
                 padding='valid'))

#model.add(Conv2DTranspose(32, kernel_size=3, padding="same"))
#model.add(UpSampling2D(3))

model.summary()
model.compile(optimizer='adam',
              loss='mse')

# =============================================================================
# Train test split
# =============================================================================
# X_train, X_test, y_train, y_test = train_test_split(X, 
#                                                     y, 
#                                                     test_size=0.2, 
#                                                     random_state=42)


# =============================================================================
# Train model
# =============================================================================


# history = model.fit(X_train,y_train,
#                     validation_data=(X_test,y_test),
#                     epochs=200, batch_size=10)

