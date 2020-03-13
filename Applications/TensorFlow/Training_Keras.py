#!/usr/bin/env python
#
# Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#   http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# @file	Training_Keras.py
# @date	13 March 2020
# @brief	This is Simple Classification Example using Keras
# @see		https://github.sec.samsung.net/AIP/nntrainer
# @author	Jijoong Moon <jijoong.moon@samsung.com>
# @bug		No known bugs except for NYI items
#
#  mobilenetv2 output (features) using cifar10 data : 10 Classes x 100 image each --> trainingSet.dat
#
#        * trainingSet.dat                       * ValSet.dat
#               62720                                 62720
#          +--------------+                      +--------------+
#          |              |                      |              |
#  10x100  |              |                10x10 |              |
#          |              |                      |              |
#          |              |                      |              |
#          +--------------+                      +--------------+
#
#   - InputFeatures(62720) ---> 10 hidden Fully Connected Layer ---> 10 Classification
#

import random
import struct
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import Callback

TOTAL_TRAIN_DATA_SIZE=100
TOTAL_LABEL_SIZE=10
TOTAL_VAL_DATA_SIZE=10
FEATURE_SIZE=62720
batch_size =32
Learning = True
Test = False
num_epoch = 10

##
# @brief Callback to get the data from model during trining
# @param[in] Callback Keras Callback object
class History_LAW(Callback):
    def on_train_begin(self, logs={}):
        self.epoch=[]
        self.weights =[]
        self.history ={}
        self.weights.append(self.model.layers[0].get_weights())
    
    def on_epoch_end(self, epoch, logs={}):
        self.weights.append(self.model.layers[0].get_weights())

##
# @brief input data generater with batch_size
#        ( 32 x 62720 :training & 32 x 10 : labels )
# @param[in] x_data : total input data
# @param[in] y_data : total label data
# @param[in] batch_size : batch_size
# @return (x_batch, y_batch)
def datagen( x_data, y_data, batch_size):
    size=len(x_data)
    while True:
        idx = np.random.permutation(size)
        x_data=x_data[idx]
        y_data=y_data[idx]

        for i in range(size // batch_size):
            x_batch = x_data[i*batch_size: (i+1)*batch_size]
            y_batch = y_data[i*batch_size: (i+1)*batch_size]

            yield x_batch, y_batch
            
##
# @brief load input data from file
# @return (InputVector, InputLabel, Validation Vector, ValidationLabel)
def load_data():
    tf.compat.v1.set_random_seed(777)

    data_size = TOTAL_TRAIN_DATA_SIZE;
    InputVector = np.zeros((TOTAL_LABEL_SIZE*data_size, FEATURE_SIZE),dtype=np.float32)
    InputLabel = np.zeros((TOTAL_LABEL_SIZE*data_size, TOTAL_LABEL_SIZE),dtype=np.float32)

    ValVector = np.zeros((TOTAL_LABEL_SIZE*TOTAL_VAL_DATA_SIZE,FEATURE_SIZE),dtype=np.float32)
    ValLabel = np.zeros((TOTAL_LABEL_SIZE*TOTAL_VAL_DATA_SIZE, TOTAL_LABEL_SIZE),dtype=np.float32)

    #read Input & Label    

    fin = open('valSet.dat','rb')
    for i in range(TOTAL_LABEL_SIZE*TOTAL_VAL_DATA_SIZE):
        for j in range(FEATURE_SIZE):
            data_str = fin.read(4)
            ValVector[i,j] = struct.unpack('f',data_str)[0]
        for j in range(TOTAL_LABEL_SIZE):
            data_str = fin.read(4)
            ValLabel[i,j] = struct.unpack('f',data_str)[0]
    fin.close()
            
    fin=open('trainingSet.dat','rb')
    for i in range(TOTAL_LABEL_SIZE*data_size):
        for j in range(FEATURE_SIZE):
            data_str = fin.read(4)
            InputVector[i,j] = struct.unpack('f',data_str)[0]
            
        for j in range(TOTAL_LABEL_SIZE):
            data_str = fin.read(4)
            InputLabel[i,j] = struct.unpack('f',data_str)[0]
    fin.close()

    return InputVector, InputLabel, ValVector, ValLabel

##
# @brief training loop
#        - epoches : 10
#        - Optimizer : Adam
#        - Activation : softmax
#        - loss : cross entropy
#
def train_nntrainer():
    InVec, InLabel, ValVec, ValLabel = load_data()    

    print('reading is done')
    inputs = tf.placeholder(tf.float32, [None, FEATURE_SIZE], name="input_X")
    labels = tf.placeholder(tf.float32,[None, TOTAL_LABEL_SIZE], name = "label")

    model = models.Sequential()
    model.add(Input(shape=(FEATURE_SIZE,)))
    
    model.add(layers.Dense(10,activation='softmax',))

    model.compile(optimizer = optimizers.Adam(lr=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    model_Hist = History_LAW()

    history = model.fit(datagen(InVec, InLabel, batch_size), epochs=10, steps_per_epoch=len(InVec)//batch_size, validation_data=datagen(ValVec, ValLabel, batch_size), validation_steps=len(ValVec)//batch_size, callbacks=[model_Hist])
    
    count =0
    
    if not os.path.exists('./nntrainer_tfmodel'):
        os.mkdir('./nntrainer_tfmodel')
        model.save('./nntrainer_tfmodel/nntrainer_keras.h5')

    score = model.evaluate(ValVec, ValLabel, verbose=1)
    print("%s: %.5f%%" % (model.metrics_names[1], score[1]*100))

##
# @brief validation loop
def validation():
    ValVector = np.zeros((TOTAL_LABEL_SIZE*TOTAL_VAL_DATA_SIZE,FEATURE_SIZE),dtype=np.float)
    ValLabel = np.zeros((TOTAL_LABEL_SIZE*TOTAL_VAL_DATA_SIZE, TOTAL_LABEL_SIZE),dtype=np.float)

    fin = open('valSet.dat','rb')
    for i in range(TOTAL_LABEL_SIZE*TOTAL_VAL_DATA_SIZE):
        for j in range(FEATURE_SIZE):
            data_str = fin.read(4)
            ValVector[i,j] = struct.unpack('f',data_str)[0]
        for j in range(TOTAL_LABEL_SIZE):
            data_str = fin.read(4)
            ValLabel[i,j] = struct.unpack('f',data_str)[0]
    fin.close()
    saved_model = tf.keras.models.load_model('./nntrainer_tfmodel/nntrainer_keras.h5')
    saved_model.summary()
    
    score = saved_model.evaluate(ValVector, ValLabel, verbose=0)
    print("%s: %.5f%%" % (saved_model.metrics_names[1], score[1]*100))
    
##
# @brief main loop
            
if __name__ == "__main__":
    if Learning:
        train_nntrainer()
    if Test:
        validation()
        
