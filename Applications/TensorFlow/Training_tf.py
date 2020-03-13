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
# @file	Training_tf.py
# @date	13 March 2020
# @brief	This is Simple Classification Example using tensorflow
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

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import struct
import os

TOTAL_TRAIN_DATA_SIZE=100
TOTAL_LABEL_SIZE=10
TOTAL_VAL_DATA_SIZE=10
FEATURE_SIZE=62720

training_epochs = 10
batch_size = 32

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
# @brief input data generater with batch_size
#        ( 32 x 62720 :training & 32 x 10 : labels )
# @param[in] x_data : total input data
# @param[in] y_data : total label data
# @param[in] batch_size : batch_size
# @return (x_batch, y_batch)
def datagen( x_data, y_data, batch_size):
    size=len(x_data)

    idx = np.random.permutation(size)
    x_data=x_data[idx]
    y_data=y_data[idx]

    for i in range(size // batch_size):
        x_batch = x_data[i*batch_size: (i+1)*batch_size]
        y_batch = y_data[i*batch_size: (i+1)*batch_size]
        yield x_batch, y_batch

##
# @brief training loop
#        - epoches : 10
#        - Optimizer : Adam
#        - Activation : softmax
#        - loss : cross entropy
def train_tensorflow():
    InVec, InLabel, ValVec, ValLabel = load_data()


    initializer = tf.contrib.layers.xavier_initializer()
    
    inputs = tf.placeholder(tf.float32, [None, FEATURE_SIZE], name="input_X")
    labels = tf.placeholder(tf.float32,[None, TOTAL_LABEL_SIZE], name = "label")
        
    W = tf.Variable(initializer([FEATURE_SIZE, TOTAL_LABEL_SIZE]), name="Weight")
    B = tf.Variable(tf.zeros([TOTAL_LABEL_SIZE]), name="Bias")

    logits = tf.matmul(inputs, W)+B
    
    hypothesis = tf.nn.softmax(logits)

    loss_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    prediction = tf.argmax(hypothesis, 1)
    correct_prediction = tf.equal(prediction, tf.argmax(labels,1))

    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost=0
            count = 0
            genTrainData=datagen(InVec, InLabel, batch_size)
            for x_batch, y_batch in genTrainData:
                c,_=sess.run([loss, optimizer],feed_dict = {inputs:x_batch, labels:y_batch})
                avg_cost+=c
                count += 1
            
            avg_cost = avg_cost/count
            
            saver.save(sess, 'saved_model/model.ckpt')
            
            count =0
            train_accuracy = 0.0
            genValidationData = datagen(ValVec, ValLabel, batch_size)
            for x_batch, y_batch in genValidationData:
                train_accuracy+=accuracy.eval(feed_dict={inputs:x_batch, labels:y_batch})
                count += 1
                
            total_acc = train_accuracy/count*100.0

            print ('Epoch:', '%06d' % (epoch+1), 'loss =', '{:.9f}'.format(avg_cost), 'Accuracy:', '{:.9f}'.format(total_acc))
            
##
# @brief main loop
if __name__ == "__main__":
    train_tensorflow()
    
