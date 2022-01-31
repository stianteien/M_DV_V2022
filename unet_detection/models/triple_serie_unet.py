# Make unet 20.01.22 Stian Teien

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Conv2DTranspose, concatenate
import numpy as np
import tensorflow as tf
#from tensorflow.image import resize

from models.vanilla_unet import vanilla_unet

class triple_unet(vanilla_unet):
    """
    Version of U-Net with dropout and size preservation (padding= 'same')
    """ 
    def __init__(self, seed=None):
        super().__init__(seed)
        


    def get_unet(self, input_img1, input_img2, n_filters = 16, dropout = 0.1,
                 batchnorm = True, n_classes = 2, last_activation="sigmoid"):

        if self.seed:
            tf.random.set_seed(self.seed)

        m1 = input_img1
        m2 = input_img2

        # Contracting Path 1
        c1 = self.conv2d_block(m1, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2), name="first_pole_1")(c1)
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
        
        # Expansive Path 1
        strides = (2,2)
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = strides, padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = strides, padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = strides, padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        

        # Contracting Path 2

        m3 = self.rationalConvTransposed(m2, start=56,end=64)

        c1_2 = self.conv2d_block(m3, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1_2 = MaxPooling2D((2, 2), name="first_pole_2")(c1_2)
        p1_2 = Dropout(dropout)(p1_2)
        #c1_22 = self.rationalConvTransposed(p1_2, start=1,end=2)
        
        c2_2 = self.conv2d_block(p1_2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2_2 = MaxPooling2D((2, 2))(c2_2)
        p2_2 = Dropout(dropout)(p2_2)
        
        c3_2 = self.conv2d_block(p2_2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3_2 = MaxPooling2D((2, 2))(c3_2)
        p3_2 = Dropout(dropout)(p3_2)
        
        c4_2 = self.conv2d_block(p3_2, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4_2 = MaxPooling2D((2, 2))(c4_2)
        p4_2 = Dropout(dropout)(p4_2)
        
        c5_2 = self.conv2d_block(p4_2, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
        
        # Expansive Path 2
        strides = (2,2)
        u6_2 = Conv2DTranspose(n_filters * 8, (3, 3), strides = strides, padding = 'same')(c5_2)
        u6_2 = concatenate([u6_2, c4_2])
        u6_2 = Dropout(dropout)(u6_2)
        c6_2 = self.conv2d_block(u6_2, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7_2 = Conv2DTranspose(n_filters * 4, (3, 3), strides = strides, padding = 'same')(c6_2)
        u7_2 = concatenate([u7_2, c3_2])
        u7_2 = Dropout(dropout)(u7_2)
        c7_2 = self.conv2d_block(u7_2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8_2 = Conv2DTranspose(n_filters * 2, (3, 3), strides = strides, padding = 'same')(c7_2)
        u8_2 = concatenate([u8_2, c2_2])
        u8_2 = Dropout(dropout)(u8_2)
        c8_2 = self.conv2d_block(u8_2, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        u9_2 = Conv2DTranspose(n_filters * 1, (3, 3), strides = strides, padding = 'same')(c8_2)
        u9_2 = concatenate([u9_2, c1_2])
        u9_2 = Dropout(dropout)(u9_2)
        c9_2 = self.conv2d_block(u9_2, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

        # Come togheter
        r2 = concatenate([c8, c9_2])

        # U 1, end of it
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = strides, padding = 'same')(r2)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

        c10 = Conv2D(n_classes, (1, 1), activation=last_activation)(c9)

        # Add inn img1 again
        c11 = concatenate([c10, m1])

        # Third
        # Contracting Path 3
        c1 = self.conv2d_block(c11, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2), name="first_pole_3")(c1)
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
        
        # Expansive Path 3
        strides = (2,2)
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = strides, padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        
        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = strides, padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        
        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = strides, padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        
        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = strides, padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

        
        outputs = Conv2D(n_classes, (1, 1), activation=last_activation)(c9)
        model = Model(inputs=[m1,m2], outputs=[outputs])
        return model