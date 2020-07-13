#!/usr/bin/env python
#
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
#
# SPDX-License-Identifier: Apache-2.0-only
#
# @file	mnist_Keras.py
# @date	13 July 2020
# @brief	This is Simple Classification Example using Keras
# @see		https://github.com/nnstreamer/nntrainer
# @author	Jijoong Moon <jijoong.moon@samsung.com>
# @bug		No known bugs except for NYI items
#
# mnist example
# inputlayer -> conv2d (5x5) 6 filters -> pooling2d 2x2 (valid) -> conv2d (5x5) 12 filters ->
#               pooling2d 2x2 (valid) -> flatten -> fully connected layer -> 10 class
#

import random
import struct
import os
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import tensorflow as tf
import numpy as np
import dataset as dataset
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import initializers

SEED=1
tf.compat.v1.reset_default_graph()
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)

batch_size =32
Learning = True
Test = False
num_epoch = 1500

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
#        ( 32 x 28 x 28 x 1 :training & 32 x 10 : labels )
# @param[in] x_data : total input data
# @param[in] y_data : total label data
# @param[in] batch_size : batch_size
# @return (x_batch, y_batch)
def datagen( x_data, y_data, batch_size):
    size=len(x_data)
    while True:
        for i in range(size // batch_size):
            x_batch = x_data[i*batch_size: (i+1)*batch_size]
            x_batch=np.reshape(x_batch, (batch_size, 1, 28,28))
            x_batch=np.transpose(x_batch, [0,2,3,1])
            y_batch = y_data[i*batch_size: (i+1)*batch_size]

            yield x_batch, y_batch
            
##
# @brief training loop
#        - epoches : 1500
#        - Optimizer : Adam
#        - Activation : softmax
#        - loss : cross entropy
#
def train_nntrainer():
    train_data_size, val_data_size, label_size, feature_size = dataset.get_data_info()
    InVec, InLabel, ValVec, ValLabel = dataset.load_data()

    print('reading is done')
    inputs = tf.placeholder(tf.float32, [None, feature_size], name="input_X")
    labels = tf.placeholder(tf.float32,[None, label_size], name = "label")

    model = models.Sequential()
    model.add(Conv2D(6, (5,5), padding='valid', activation='sigmoid', input_shape=(28,28,1), kernel_initializer=initializers.Zeros(), bias_initializer=initializers.Zeros()))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Conv2D(12, (5,5), padding='valid', activation='sigmoid', kernel_initializer=initializers.Zeros(), bias_initializer=initializers.Zeros()))
    model.add(AveragePooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(layers.Dense(10,activation='softmax',kernel_initializer=initializers.Zeros(), bias_initializer=initializers.Zeros()))

    model.compile(optimizer = optimizers.Adam(lr=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    model.summary()

    model_Hist = History_LAW()

    history = model.fit(datagen(InVec, InLabel, batch_size), epochs=num_epoch, steps_per_epoch=len(InVec)//batch_size, validation_data=datagen(ValVec, ValLabel, batch_size), validation_steps=len(ValVec)//batch_size, callbacks=[model_Hist])
    
    count =0
    
    if not os.path.exists('./nntrainer_tfmodel'):
        os.mkdir('./nntrainer_tfmodel')
        model.save('./nntrainer_tfmodel/nntrainer_keras.h5')

    ValVec = np.reshape(ValVec, (label_size*val_data_size, 1,28,28))
    ValVec = np.transpose(ValVec, [0,2,3,1])
    score = model.evaluate(ValVec, ValLabel, verbose=1)
    print("%s: %.5f%%" % (model.metrics_names[1], score[1]*100))

##
# @brief validation loop
def validation():
    ValVector = np.zeros((label_size*val_data_size,feature_size),dtype=np.float)
    ValLabel = np.zeros((label_size*val_data_size, label_size),dtype=np.float)

    fin = open('mnist_trainingSet.dat','rb')
    for i in range(label_size*val_data_size):
        for j in range(feature_size):
            data_str = fin.read(4)
            ValVector[i,j] = struct.unpack('f',data_str)[0]
        for j in range(label_size):
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
        
