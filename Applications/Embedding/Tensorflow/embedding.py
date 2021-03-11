#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
#
# @file	embedding.py
# @date	15 March 2021
# @brief	This is Simple Embedding Layer Training Example
# @see		https://github.com/nnstreamer/nntrainer
# @author	Jijoong Moon <jijoong.moon@samsung.com>
# @bug		No known bugs except for NYI items
#
#

from numpy import array
from numpy import zeros
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import models, layers, optimizers
import os.path

##
# @brief Save Tensor Data
def save(filename, *data):
    with open(filename, 'ab+') as outfile:
        print(data[0][0])
        for item in data:
          np.array(item, dtype=np.float32).tofile(outfile)
          try:
            print(item.shape, " data is generated")
            print(item)
          except:
            pass
##
# @brief Generate Model
def create_model(in_dim, out_dim, input_length, trainable=True):
        model = models.Sequential()
        model.add(layers.Embedding(in_dim, out_dim, input_length=input_length, trainable=trainable))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

##
# @brief Generate Input
def gen_input(outfile='embedding_input.txt'):
        max_length = 4
        docs = ['Well done!',
		'Good work',
		'Great effort',
		'nice work',
		'Excellent!',
		'Weak',
		'Poor effort!',
		'not good',
		'poor work',
		'Could have done better.']

        labels = array([1,1,1,1,1,0,0,0,0,0])
        t = Tokenizer()
        t.fit_on_texts(docs)
        vocab_size = len(t.word_index) + 1 # 15
        encoded_docs = t.texts_to_sequences(docs)
        padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

        if not os.path.exists(outfile):
                f=open(outfile,'w')
                for i, doc in enumerate(padded_docs):
                        for index in doc:
                                f.write(str(index)+" ")
                        f.write(str(labels[i]) + "\n")
                f.close()

        return padded_docs, labels, max_length, vocab_size


out_dim = 8
##
# @brief Train Embedding Layer
def train_nntrainer():

        padded_docs, labels, max_length, vocab_size = gen_input('embedding_input.txt')

        model = create_model(vocab_size, out_dim, max_length)

        optimizer = optimizers.Adam(learning_rate=1.0e-4, beta_1=0.9, beta_2=0.999, epsilon=1.0e-7)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

        print(model.summary())

        save("model.bin", model.get_weights()[0]) # embedding weight
        save("model.bin", model.get_weights()[1]) # fc weight
        save("model.bin", model.get_weights()[2]) # fc bias
        epoch = [1]
        iter = [0]
        save("model.bin", epoch)
        save("model.bin", iter)

        save ("embedding_weight_input.in", model.get_weights()[0])

        model.fit(padded_docs, labels, epochs=1)
        save ("embedding_weight_golden.out", model.get_weights()[0])
        save ("fc_weight_golden.out", model.get_weights()[1])

        loss, accuracy = model.evaluate(padded_docs, labels)
        print('Accuracy: %f' % (accuracy*100))

##
# @brief main loop
if __name__ == "__main__":
        train_nntrainer()
