# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
#
# @file   main.cpp
# @date   30 Jan 2023
# @see    https://github.com/nnstreamer/nntrainer
# @author Seungbaek Hong <sb92.hong@samsung.com>
# @bug	  No known bugs except for NYI items
# @brief  This is LSTM Example for Tensorflow (only training with dummy data)

import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")

seed = 0
tf.random.set_seed(seed)
tf.config.threading.set_intra_op_parallelism_threads(1)

EPOCH = 10
DB_SIZE = 64
BATCH_SIZE = 64
IMG_SIZE = 224*224*3
OUTPUT_SIZE = 10


class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = tf.keras.layers.LSTM(OUTPUT_SIZE, batch_input_shape = (BATCH_SIZE, 1, IMG_SIZE), return_sequences=True)

    def call(self, x):
        output = self.lstm(x)
        return output


def train(dataloader, model, loss_fn, optimizer):
    epoch_loss, num_of_batch = 0, len(dataloader)

    for X_batch, y_batch in dataloader:
        # Compute prediction error
        with tf.GradientTape() as tape:
            pred = model(X_batch)            
            loss = loss_fn(pred, y_batch)

        # Backpropagation
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        epoch_loss += loss / num_of_batch

    return epoch_loss

def make_dummy_database(num_of_samples):
    X = tf.random.normal((num_of_samples, 1, IMG_SIZE))
    y = tf.random.normal((num_of_samples, 1, OUTPUT_SIZE))
    return X, y


if __name__ == '__main__':
    model = LSTM()
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.SGD()

    for epoch in range(EPOCH):
        X_train, y_train = make_dummy_database(DB_SIZE)
        dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
        epoch_loss = train(dataset, model, loss_fn, optimizer)
        print(f"loss: {epoch_loss:>7f}  [{epoch+1:>5d}/{EPOCH}]")
        del X_train, y_train, dataset, epoch_loss
