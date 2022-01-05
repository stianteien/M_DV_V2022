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

from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, \
                         UpSampling2D, Conv2DTranspose, MaxPooling2D



# =============================================================================
# Load data
# =============================================================================

img_70 = np.load("E:/M-DV-STeien/Sharping_test/2019_04_vnir70cm_utdrag.npy")
img_30 = np.load("E:/M-DV-STeien/Sharping_test/2019_04_vnir30cm_utdrag.npy")


# =============================================================================
# Reduce to fitting size and use only RGB
# =============================================================================
img_30_rgb = np.dstack((img_30[:,:,76],
                        img_30[:,:,46],
                        img_30[:,:,21]))
img_70_rgb = np.dstack((img_70[:,:,76],
                        img_70[:,:,46],
                        img_70[:,:,21]))

img_30_gray = img_30[:,:,76]
img_70_gray = img_70[:,:,76]


# =============================================================================
# Extract n amount of images from image
# =============================================================================

X_shape = 60
y_shape = 140
n = 100

X_r,X_c = img_70_gray.shape
y_r,y_c = img_30_gray.shape


X = []
y = []

for _ in range(n):
    a,b = (np.random.randint(0,X_r-X_shape), np.random.randint(0,X_c-X_shape))
    c,d = (np.random.randint(0,y_r-y_shape), np.random.randint(0,y_c-y_shape))
    
    X.append(img_70_gray[a:a+X_shape, b:b+X_shape].reshape(X_shape,X_shape,1))  
    y.append(img_30_gray[c:c+y_shape, d:d+y_shape].reshape(y_shape,y_shape,1))

X = np.array(X)
y = np.array(y)


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
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42)


# =============================================================================
# Train model
# =============================================================================


#history = model.fit(X_train,y_train,
#                    validation_data=(X_test,y_test),
#                    epochs=200, batch_size=10)

