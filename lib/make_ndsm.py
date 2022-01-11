# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 10:51:06 2022

@author: Stian
"""

from PIL import Image, ImageChops
from PIL.TiffTags import TAGS
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i_DSM", help="Input path for DSM file", required=True)
parser.add_argument("-i_DTM", help="Input path for DTM file",required=True)
parser.add_argument("-o", help="Output file",required=True)

args = parser.parse_args()


DSM = Image.open(args.i_DSM)
DTM = Image.open(args.i_DTM)

#nDSM = ImageChops.subtract(DSM,DTM, scale=1, offset=0)
#meta_dict = {TAGS[key] : DSM.tag[key] for key in DSM.tag.keys()}

nDSM =  np.asarray(DSM) -  np.asarray(DTM)
nDSM[nDSM>200] = 0
nDSM[nDSM<=0] = 0
im = Image.fromarray(nDSM)
im.save(args.o,
        tiffinfo=DSM.tag, save_all=True)