#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
#
# @file	vgg_keras.py
# @date	08 Oct 2020
# @brief	This is VGG Example using Keras
# @see		https://github.com/nnstreamer/nntrainer
# @author	Jijoong Moon <jijoong.moon@samsung.com>
# @bug		No known bugs except for NYI items
#
# vgg example : filter size is reduced.
# conv3_16 - conv3_16, max_pooling, conv3_32 - conv3_32, max_pooling, conv3_64 - conv3_64 - conv3_64, max_pooling,
# conv3_128 - conv3_128 - conv3_128, max_pooling, conv3_128 - conv3_128 - conv3_128, fc_128, fc_128, fc_100, softmax
#

import random
import struct
import os
import sys
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import tensorflow as tf
import numpy as np
import dataset as dataset
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import initializers
from tensorflow.keras.models import load_model

np.set_printoptions(threshold=sys.maxsize)
SEED=1

tf.compat.v1.reset_default_graph()
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)

batch_size =128
Learning = True
Test = False
num_epoch = 1500
DEBUG = True
USE_FIT = False

def save(filename, *data):
    with open(filename, 'ab+') as outfile:
        for item in data:
          np.array(item, dtype=np.float32).tofile(outfile)
          try:
            print(item.shape, " data is generated")
          except:
            pass

##
# @brief input data generater with batch_size
#        ( 128 x 32 x 32 x 3 :training & 128 x 1000 : labels )
# @param[in] x_data : total input data
# @param[in] y_data : total label data
# @param[in] batch_size : batch_size
# @return (x_batch, y_batch)
def datagen(x_data, y_data, batch_size):
    size=len(x_data)
    while True:
        for i in range(size // batch_size):
            x_batch = x_data[i*batch_size: (i+1)*batch_size]
            x_batch = np.reshape(x_batch, (batch_size, 3, 32,32))
            x_batch = np.transpose(x_batch, [0,2,3,1])
            y_batch = y_data[i*batch_size: (i+1)*batch_size]
            yield x_batch, y_batch

def create_model():
    model = models.Sequential()
    model.add(tf.keras.Input(shape=(32, 32, 3)))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(16, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(64, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(128, (3,3), padding='same', activation='relu', bias_initializer=initializers.Zeros()))
    model.add(Conv2D(128, (3,3), padding='same', bias_initializer=initializers.Zeros()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(layers.Dense(128, bias_initializer=initializers.Zeros()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.Dense(128, bias_initializer=initializers.Zeros()))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.Dense(100, bias_initializer=initializers.Zeros()))
    return model

##
# @brief training loop
#        - epochs : 1500
#        - Optimizer : Adam
#        - Activation : softmax
#        - loss : cross entropy
#
def train_nntrainer(target):
    train_data_size, val_data_size, label_size, feature_size = dataset.get_data_info(target)
    InVec, InLabel, ValVec, ValLabel = dataset.load_data(target)

    model = create_model()
    model.summary()

    if USE_FIT == False:
        ## Method 1 : using tensorflow session (training and evaluating manually)
        inputs = tf.placeholder(tf.float32, [None, 32,32,3], name="input_X")
        labels = tf.placeholder(tf.float32,[None, 100], name = "label")
        sess=tf.compat.v1.Session()

        tf_logit = model(inputs, training=True)
        tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=tf_logit))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-2, decay_steps=10000, decay_rate=0.96)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1.0e-7, beta_1=0.9, beta_2=0.999)

        trainable_variables = tf.compat.v1.trainable_variables()
        tf_grad = optimizer.get_gradients(tf_loss, params = trainable_variables)
        train_op = optimizer.apply_gradients(zip(tf_grad, trainable_variables))
        var_to_run = [train_op, tf_loss, tf.reduce_sum(tf.cast(tf.equal(tf.math.argmax(tf.nn.softmax(tf_logit), axis=1), tf.math.argmax(labels, axis=1)), tf.float32))/batch_size]

        tf_logit_eval = model(inputs, training=False)
        tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=tf_logit_eval))
        tf_logit_eval = tf.nn.softmax(tf_logit_eval)
        tf_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.math.argmax(tf_logit_eval, axis=1), tf.math.argmax(labels, axis=1)), tf.float32))/batch_size
        infer_to_run = [tf_accuracy, tf_loss]

        sess.run(tf.compat.v1.global_variables_initializer())

        conv2_0 = np.transpose(model.get_weights()[0], [3,2,0,1])
        conv2_1 = np.transpose(model.get_weights()[2], [3,2,0,1])
        conv2_2 = np.transpose(model.get_weights()[4], [3,2,0,1])
        conv2_3 = np.transpose(model.get_weights()[6], [3,2,0,1])
        conv2_4 = np.transpose(model.get_weights()[8], [3,2,0,1])
        conv2_5 = np.transpose(model.get_weights()[10], [3,2,0,1])
        conv2_6 = np.transpose(model.get_weights()[12], [3,2,0,1])
        conv2_7 = np.transpose(model.get_weights()[14], [3,2,0,1])
        conv2_8 = np.transpose(model.get_weights()[16], [3,2,0,1])
        conv2_9 = np.transpose(model.get_weights()[18], [3,2,0,1])
        conv2_10 = np.transpose(model.get_weights()[20], [3,2,0,1])
        conv2_11 = np.transpose(model.get_weights()[22], [3,2,0,1])
        conv2_12 = np.transpose(model.get_weights()[24], [3,2,0,1])

        bn_1_0 = np.transpose(model.get_weights()[26])
        bn_1_1 = np.transpose(model.get_weights()[27])
        bn_1_2 = np.transpose(model.get_weights()[28])
        bn_1_3 = np.transpose(model.get_weights()[29])

        fc_0_0 = np.transpose(model.get_weights()[30])
        fc_0_1 = np.transpose(model.get_weights()[31])

        bn_2_0 = np.transpose(model.get_weights()[32])
        bn_2_1 = np.transpose(model.get_weights()[33])
        bn_2_2 = np.transpose(model.get_weights()[34])
        bn_2_3 = np.transpose(model.get_weights()[35])

        fc_1_0 = np.transpose(model.get_weights()[36])
        fc_1_1 = np.transpose(model.get_weights()[37])

        bn_3_0 = np.transpose(model.get_weights()[38])
        bn_3_1 = np.transpose(model.get_weights()[39])
        bn_3_2 = np.transpose(model.get_weights()[40])
        bn_3_3 = np.transpose(model.get_weights()[41])

        fc_2_0 = np.transpose(model.get_weights()[42])
        fc_2_1 = np.transpose(model.get_weights()[43])

        save("model.bin", conv2_0)
        save("model.bin", model.get_weights()[1])
        save("model.bin", conv2_1)
        save("model.bin", model.get_weights()[3])
        save("model.bin", conv2_2)
        save("model.bin", model.get_weights()[5])
        save("model.bin", conv2_3)
        save("model.bin", model.get_weights()[7])
        save("model.bin", conv2_4)
        save("model.bin", model.get_weights()[9])
        save("model.bin", conv2_5)
        save("model.bin", model.get_weights()[11])
        save("model.bin", conv2_6)
        save("model.bin", model.get_weights()[13])
        save("model.bin", conv2_7)
        save("model.bin", model.get_weights()[15])
        save("model.bin", conv2_8)
        save("model.bin", model.get_weights()[17])
        save("model.bin", conv2_9)
        save("model.bin", model.get_weights()[19])
        save("model.bin", conv2_10)
        save("model.bin", model.get_weights()[21])
        save("model.bin", conv2_11)
        save("model.bin", model.get_weights()[23])
        save("model.bin", conv2_12)
        save("model.bin", model.get_weights()[25])

        save("model.bin", bn_1_0)
        save("model.bin", bn_1_1)
        save("model.bin", bn_1_2)
        save("model.bin", bn_1_3)
        save("model.bin", fc_0_0)
        save("model.bin", fc_0_1)
        save("model.bin", bn_2_0)
        save("model.bin", bn_2_1)
        save("model.bin", bn_2_2)
        save("model.bin", bn_2_3)
        save("model.bin", fc_1_0)
        save("model.bin", fc_1_1)
        save("model.bin", bn_3_0)
        save("model.bin", bn_3_1)
        save("model.bin", bn_3_2)
        save("model.bin", bn_3_3)
        save("model.bin", fc_2_0)
        save("model.bin", fc_2_1)

        for i in range(0, num_epoch):
            count = 0
            accuracy = 0;
            loss = 0
            for x, y in datagen(InVec, InLabel, batch_size):
                feed_dict = {inputs: x, labels: y}
                tf_out = sess.run(var_to_run, feed_dict = feed_dict)
                loss += tf_out[1]
                accuracy += tf_out[2]
                count = count + 1
                if count == len(InVec) // batch_size:
                    break;

            training_accuracy = (accuracy / count) * 100.0
            training_loss = loss/count

            count = 0
            accuracy = 0;
            loss = 0;
            for x, y in datagen(ValVec, ValLabel, batch_size):
                feed_dict = {inputs: x, labels: y}
                infer_out = sess.run(infer_to_run, feed_dict = feed_dict)
                accuracy += infer_out[0]
                loss += infer_out[1]
                count = count + 1
                if count == len(ValVec) // batch_size:
                    break;

            accuracy = (accuracy / count) * 100.0
            loss = loss / count
            print('#{}/{} - Training Loss: {:10.6f} - Training Accuracy: {:10.6f} >> [ Accuracy: {:10.6f}% - Validation Loss : {:10.6f} ]'. format(i + 1, num_epoch, training_loss, training_accuracy, accuracy, loss))
    else:
        ## Method 1 : using keras fit (training and evaluating manually)
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 1e-2, decay_steps=10000, decay_rate=0.96)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1.0e-7, beta_1=0.9, beta_2=0.999)

        model.compile(optimizer = optimizer,
                      loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])
        model.fit(datagen(InVec, InLabel, batch_size),
                  epochs = num_epoch,
                  steps_per_epoch = len(InVec) // batch_size,
                  validation_data = datagen(ValVec, ValLabel, batch_size),
                  validation_steps = len(ValVec) // batch_size,
                  shuffle = False)

##
# @brief main loop

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "train"
    target1 = sys.argv[2] if len(sys.argv) > 2 else "train"

    if target1 == "train":
        train_nntrainer(target)

