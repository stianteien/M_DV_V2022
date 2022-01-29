# Make unet 20.01.22 Stian Teien

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
import numpy as np
import tensorflow as tf
#from tensorflow.image import resize

from unet_detection.models.vanilla_unet import vanilla_unet


class unet_2input_deep(vanilla_unet):
    """
    Version of U-Net with dropout and size preservation (padding= 'same')
    """ 
    def __init__(self, seed=None):
        super().__init__(seed)
        #self.seed = seed


    def get_unet(self, input_img1, input_img2, n_filters = 16, dropout = 0.1,
                 batchnorm = True, n_classes = 2, last_activation="sigmoid"):
        
        if self.seed:
            tf.random.set_seed(self.seed)

        m1 = input_img1
        m2 = input_img2

        # Contracting Path
        c1 = self.conv2d_block(m1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        # Add other input here and concanate
        m3 = self.rationalConvTransposed(m2, start=56,end=64)
        m4 = concatenate([p1, m3])

        c2 = self.conv2d_block(m4, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)
        
        c3 = self.conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)
        
        c4 = self.conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)
        
        c5 = self.conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
        
        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        
        outputs = Conv2D(n_classes, (1, 1), activation=last_activation)(c9)
        model = Model(inputs=[m1, m2], outputs=[outputs])
        return model
