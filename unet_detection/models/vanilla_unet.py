# Make unet 20.01.22 Stian Teien

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import backend as K
#from tensorflow.image import resize


class vanilla_unet:
    
    def __init__(self, seed=None):
        self.seed = seed

    def jacard(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum ( y_true_f * y_pred_f)
        union = K.sum ( y_true_f + y_pred_f - y_true_f * y_pred_f)
        return intersection/union

    def rationalConv(self, x, filters=16, kernel_size=(3,3), padding="same", start=7, end=3):
        """
        Where ordinary convolutions down-scale by an integer stride, 
        the rational version up-samples and down-scales an integer stride to 
        reach a desired rational upscaling.

        Example: 7 -> 3 => 7 * bilinear(9/7) = 9, 9 / conv(3) = 3
        """
        size = x.shape
        stride = np.ceil(start/end).astype(int)
        scaling = (end*stride)/start
        
        x = tf.image.resize(x, tf.constant(np.round([size[1]*scaling, size[2]*scaling]), dtype="int32"))
        if stride > 1:
            x = Conv2D(filters=filters, kernel_size=kernel_size, strides = (stride,stride), padding=padding)(x)
        return x

    def rationalConvTransposed(self, x, filters=16, kernel_size=(3,3), padding="same", start=3, end=7):
        """
        Where ordinary transposed convolutions up-scale by an integer stride, 
        the rational version up-scales an integer stride and down-samples to 
        reach a desired rational upscaling.
        
        Example: 3 -> 7 => 3 * convTr(3) = 9, 9 * bilinear(7/9) = 7
        """
        size = x.shape
        stride = np.ceil(end/start).astype(int)
        scaling = end/start
        
        if stride > 1:
            x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides = (stride,stride), padding=padding)(x)
        x = tf.image.resize(x, tf.constant(np.round([size[1]*scaling, size[2]*scaling]), dtype="int32"))
        return x
	
    
    
    """
    Version of U-Net with dropout and size preservation (padding= 'same')
    """ 
    
    def conv2d_block(self, input_tensor, n_filters, kernel_size = 3, batchnorm = True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        # second layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                kernel_initializer = 'he_normal', padding = 'same')(x)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
        return x


    def get_unet(self, input_img, im2, n_filters = 16, dropout = 0.1,
                 batchnorm = True, n_classes = 2, last_activation="sigmoid"):

        if self.seed:
            tf.random.set_seed(self.seed)

        
        # Contracting Path
        c1 = self.conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)
        
        c2 = self.conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
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
        model = Model(inputs=[input_img], outputs=[outputs])
        return model
