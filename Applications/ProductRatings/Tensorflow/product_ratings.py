#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
#
# @file	  product_ratings.py
# @date	  15 March 2021
# @brief  This is a simple recommentation system Example
# @see    https://github.com/nnstreamer/nntrainer
# @author Parichay Kapoor <pk.kapoor@samsung.com>
# @author Jijoong Moon <jijoong.moon@samsung.com>
# @bug    No known bugs except for NYI items
#
#

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense
from tensorflow.keras.models import Model

# assumes that the top row of the dataset file contains the header info
# userId, productId, rating
dataset = pd.read_csv('product_ratings.csv', sep = ',', header=[0])
n_users = dataset['userId'].max()
n_products = dataset['productId'].max()
train, test = train_test_split(dataset, test_size=0.2)

# Create the layers and model
product_input = Input(shape=[1], name="product")
product_embed = Embedding(n_products + 1, 5)(product_input)
product_out = Flatten()(product_embed)
user_input = Input(shape=[1], name="input")
user_embed = Embedding(n_users + 1, 5)(user_input)
user_out = Flatten()(user_embed)
concat_feat = Concatenate()([product_out, user_out])
fc1 = Dense(128, activation='relu')(concat_feat)
fc2 = Dense(32, activation='relu')(fc1)
out = Dense(1)(fc2)
model = Model([user_input, product_input], out)

# train and evaluate the model
model.compile('adam', 'mean_squared_error')
model.fit([train.userId, train.productId], train.rating, epochs=10, verbose=1)
model.evaluate([test.userId, test.productId], test.rating)

