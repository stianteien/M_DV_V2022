# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 11:07:26 2022

@author: Stian
"""

import matplotlib.pyplot as plt
import numpy as np
from spectral import imshow, ndvi
from cv2 import Sobel, Laplacian, watershed
import cv2 as cv

import spectral
from PIL import Image

y = np.load("E:/M-DV-STeien/juni2021/04/temp_data/y_data.npy")
x = np.load("E:/M-DV-STeien/juni2021/04/temp_data/X_data.npy")

classes = {"None": 0,
               "ceramic": 1,
               "concrete": 2,
               "eternit": 3,
               "metal": 4,
               "tar roofing paper":5}