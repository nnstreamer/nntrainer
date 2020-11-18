// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   16 November 2020
 * @brief  This file contains the execution part of LayerClient example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <iostream>
#include <memory>

#include <model.h>

#include <pow.h>

#define BATCH_SIZE 10
#define FEATURE_SIZE 100
#define NUM_CLASS 10

/**
 * @brief      get data which size is batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int constant_generator_cb(float **outVec, float **outLabel, bool *last,
                          void *user_data) {
  static int count = 0;
  unsigned int i;
  unsigned int data_size = BATCH_SIZE * FEATURE_SIZE;

  for (i = 0; i < data_size; ++i) {
    outVec[0][i] = 1.0f;
  }

  outLabel[0][0] = 1.0f;
  for (i = 0; i < NUM_CLASS - 1; ++i) {
    outLabel[0][i] = 0.0f;
  }

  if (count == 10) {
    *last = true;
    count = 0;
  } else {
    *last = false;
    count++;
  }

  return ML_ERROR_NONE;
}

int main(int argc, char *argv[]) {
  /**< add argc */
  auto model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  model->loadFromConfig("..model.ini");

  std::cout << "This is an example scaffolding of LayerClient";
}
