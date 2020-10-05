#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
#
# @file	convert.py
# @date	12 Oct 2020
# @brief	This is script to convert cifar100 data set for vgg model
# @see		https://github.com/nnstreamer/nntrainer
# @author	Jijoong Moon <jijoong.moon@samsung.com>
# @bug		No known bugs except for NYI items
#
# python3 convert.py path_for_cifar100 [train | val]
#  - trainingSet.dat for training : 100 images x 100 classes
#  - valSet.dat for validation : 20 images x 100 classes
#

import os
import sys
import imageio
import numpy as np
import random

from PIL import Image

num_val = 20
num_train = 100
num_class = 100



##
# @brief extract_features from image
#
def extract_features(path, index, val):
    label = [ "apple",      "bridge",    "cockroach",  "hamster",      "motorcycle",
    "plain",      "seal",      "table",      "willow_tree",  "aquarium_fish",
    "bus",        "couch",     "house",      "mountain",     "plate",
    "shark",      "tank",      "wolf",       "baby",         "butterfly",
    "crab",       "kangaroo",  "mouse",      "poppy",        "shrew",
    "telephone",  "woman",     "bear",       "camel",        "crocodile",
    "keyboard",   "mushroom",  "porcupine",  "skunk",        "television",
    "worm",       "beaver",    "can",        "cup",          "lamp",
    "oak_tree",   "possum",    "skyscraper", "tiger",        "bed",
    "castle",     "dinosaur",  "lawn_mower", "orange",       "rabbit",
    "snail",      "tractor",   "bee",        "caterpillar",  "dolphin",
    "leopard",    "orchid",    "raccoon",    "snake",        "train",
    "beetle",     "cattle",    "elephant",   "lion",         "otter",
    "ray",        "spider",    "trout",      "bicycle",      "chair",
    "flatfish",   "lizard",    "palm_tree",  "road",         "squirrel",
    "tulip",      "bottle",    "chimpanzee", "forest",       "lobster",
    "pear",       "rocket",    "streetcar",  "turtle",       "bowl",
    "clock",      "fox",       "man",        "pickup_truck", "rose",
    "sunflower",  "wardrobe",  "boy",        "cloud",        "girl",
    "maple_tree", "pine_tree", "sea",        "sweet_pepper", "whale"]

    label_name = np.array(label)

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

    label = np.zeros(100, dtype=np.uint8)
    label[class_id] = 1
    return data, label

##
# @brief main loop
if __name__ == "__main__":
    path = sys.argv[1]
    val = sys.argv[2]

    if val == "val":
        filename=os.path.join("vgg_valSet.dat")
        indexs = np.zeros(num_class*num_val, dtype=np.int)
        for i in range(num_class*num_val):
            indexs[i] = i
        validation = True
    else:
        filename=os.path.join("vgg_trainingSet.dat")
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
