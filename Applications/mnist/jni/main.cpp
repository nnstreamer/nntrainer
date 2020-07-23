/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * SPDX-License-Identifier: Apache-2.0-only
 *
 * @file	main.cpp
 * @date	01 July 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is MNIST Example with
 *
 * Input 28x28
 * Conv2D 5 x 5 : 6 filters, stride 1x1, padding=0,0
 * Pooling2D : 2 x 2, Average pooling, stride 1x1
 * Conv2D 6 x 5 x 5 : 12 filters, stride 1x1, padding=0,0
 * Pooling2D : 2 x 2, Average Pooling, stride 1x1
 * Flatten
 * Fully Connected Layer with 10 unit
 *
 */

#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include "databuffer.h"
#include "databuffer_func.h"
#include "neuralnet.h"
#include "nntrainer_error.h"
#include "tensor.h"

#define TRAINING true

/**
 * @brief     Data size for each category
 */
const unsigned int total_train_data_size = 100;

const unsigned int total_val_data_size = 100;

const unsigned int total_test_data_size = 100;

const unsigned int buffer_size = 100;

/**
 * @brief     Number of category : Three
 */
const unsigned int total_label_size = 10;

unsigned int train_count = 0;
unsigned int val_count = 0;

const unsigned int mini_batch = 32;

const unsigned int feature_size = 784;

const float tolerance = 0.1;

std::string data_path;

/**
 * @brief     step function
 * @param[in] x value to be distinguished
 * @retval 0.0 or 1.0
 */
float stepFunction(float x) {
  if (x + tolerance > 1.0) {
    return 1.0;
  }

  if (x - tolerance < 0.0) {
    return 0.0;
  }

  return x;
}

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] outVec
 * @param[out] outLabel
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
bool getData(std::ifstream &F, std::vector<float> &outVec,
             std::vector<float> &outLabel, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position = (uint64_t)((feature_size + total_label_size) *
                                 (uint64_t)id * sizeof(float));

  if (position > file_length) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  for (unsigned int i = 0; i < feature_size; i++)
    F.read((char *)&outVec[i], sizeof(float));
  for (unsigned int i = 0; i < total_label_size; i++)
    F.read((char *)&outLabel[i], sizeof(float));

  return true;
}

/**
 * @brief      get data which size is mini batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getMiniBatch_train(float **outVec, float **outLabel, bool *last,
                       void *user_data) {
  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = total_train_data_size;

  std::string filename = "mnist_trainingSet.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  if (data_size * total_label_size - train_count < mini_batch) {
    *last = true;
    train_count = 0;
    return ML_ERROR_NONE;
  }

  count = 0;
  for (unsigned int i = train_count; i < train_count + mini_batch; i++) {
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(total_label_size);

    getData(F, o, l, i);

    for (unsigned int j = 0; j < feature_size; ++j)
      outVec[0][count * feature_size + j] = o[j];
    for (unsigned int j = 0; j < total_label_size; ++j)
      outLabel[0][count * total_label_size + j] = l[j];
    count++;
  }

  F.close();
  *last = false;
  train_count += mini_batch;
  return ML_ERROR_NONE;
}

/**
 * @brief      get data which size is mini batch for validation
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getMiniBatch_val(float **outVec, float **outLabel, bool *last,
                     void *user_data) {

  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = total_val_data_size;

  std::string filename = "mnist_trainingSet.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  if (data_size * total_label_size - val_count < mini_batch) {
    *last = true;
    val_count = 0;
    return ML_ERROR_NONE;
  }

  count = 0;
  for (unsigned int i = val_count; i < val_count + mini_batch; i++) {
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(total_label_size);

    getData(F, o, l, i);

    for (unsigned int j = 0; j < feature_size; ++j)
      outVec[0][count * feature_size + j] = o[j];
    for (unsigned int j = 0; j < total_label_size; ++j)
      outLabel[0][count * total_label_size + j] = l[j];
    count++;
  }

  F.close();
  *last = false;
  val_count += mini_batch;
  return ML_ERROR_NONE;
}

/**
 * @brief     create NN
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cout << "./nntrainer_mnist mnist.ini\n";
    exit(0);
  }
  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];

  srand(time(NULL));
  std::vector<std::vector<float>> inputVector, outputVector;
  std::vector<std::vector<float>> inputValVector, outputValVector;
  std::vector<std::vector<float>> inputTestVector, outputTestVector;

  /**
   * @brief     Data buffer Create & Initialization
   */
  std::shared_ptr<nntrainer::DataBufferFromCallback> DB =
    std::make_shared<nntrainer::DataBufferFromCallback>();
  DB->setFunc(nntrainer::BUF_TRAIN, getMiniBatch_train);
  DB->setFunc(nntrainer::BUF_VAL, getMiniBatch_val);

  /**
   * @brief     Neural Network Create & Initialization
   */
  nntrainer::NeuralNetwork NN;
  try {
    NN.setConfig(config);
    NN.loadFromConfig();
  } catch (...) {
    std::cerr << "Error during loadFromConfig" << std::endl;
    NN.finalize();
    return 0;
  }

  try {
    NN.init();
  } catch (...) {
    std::cerr << "Error during init" << std::endl;
    NN.finalize();
    return 0;
  }

  NN.readModel();
  NN.setDataBuffer((DB));

  /**
   * @brief     Neural Network Train & validation
   */
  try {
    NN.train();
  } catch (...) {
    std::cerr << "Error during train" << std::endl;
    NN.finalize();
    return 0;
  }

  /**
   * @brief     Finalize NN
   */
  NN.finalize();
  return 0;
}
