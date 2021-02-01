#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
#
# @file	convert.py
# @date	28 Jan 2021
# @brief	This is script to convert cifar10 data set for resnet model
# @see		https://github.com/nnstreamer/nntrainer
# @author	Jijoong Moon <jijoong.moon@samsung.com>
# @bug		No known bugs except for NYI items
#
# python3 convert.py path_for_cifar10 [train | val]
#  - trainingSet.dat for training : 100 images x 10 classes
#  - valSet.dat for validation : 20 images x 10 classes
#

import os
import sys
import imageio
import numpy as np
import random

from PIL import Image

num_val = 20
num_train = 100
num_class = 10

##
# @brief extract_features from image
#
def extract_features(path, index, val):
    label_ = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]
    
    label_name = np.array(label_)

    path = path + "/train/"
    class_id =0

    if(val == True):
        class_id = index // num_val
        path = path + (label_name[class_id])+'/'+ '{:04d}'.format(500-(index % num_val))+'.bmp'
    else:
        class_id = index // num_train
        path = path + (label_name[class_id])+'/'+ '{:04d}'.format((index % num_train)+1)+'.bmp'

    image = Image.open(path)
    data = np.array(image)
    data = np.transpose(data, [2,0,1])

    label = np.zeros(num_class, dtype=np.uint8)
    label[class_id] = 1
    return data, label

##
# @brief main loop
if __name__ == "__main__":
    path = sys.argv[1]
    val = sys.argv[2]

    if val == "val":
        filename=os.path.join("resnet_valSet.dat")
        indexs = np.zeros(num_class*num_val, dtype=np.int)
        for i in range(num_class*num_val):
            indexs[i] = i
        validation = True
    else:
        filename=os.path.join("resnet_trainingSet.dat")
        indexs = np.zeros(num_class*num_train, dtype=np.int)
        for i in range(num_class*num_train):
            indexs[i] = i
        validation = False
     
    count = 0;

    with open(filename, 'ab+') as outfile:
        for i in indexs:
            data, label = extract_features(path, i, validation)
            np.array(data, dtype=np.float32).tofile(outfile)
            np.array(label, dtype=np.float32).tofile(outfile)
            count += 1            
    outfile.close()
