// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   multi_loader.h
 * @date   5 July 2023
 * @brief  multi data loader
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "multi_loader.h"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <nntrainer_error.h>
#include <random>

namespace nntrainer::util {

namespace {
/**
 * @brief fill last to the given memory
 * @note this function increases iteration value, if last is set to true,
 * iteration resets to 0
 *
 * @param[in/out] iteration current iteration
 * @param data_size Data size
 * @return bool true if iteration has finished
 */
bool updateIteration(unsigned int &iteration, unsigned int data_size) {
  if (++iteration == data_size) {
    iteration = 0;
    return true;
  }
  return false;
};

} // namespace

MultiDataLoader::MultiDataLoader(const std::vector<TensorDim> &input_shapes,
                                 const std::vector<TensorDim> &output_shapes,
                                 int data_size_) :
  iteration(0),
  data_size(data_size_),
  count(0),
  input_shapes(input_shapes),
  output_shapes(output_shapes),
  input_dist(0, 255),
  label_dist(0, output_shapes.front().width() - 1) {
  NNTR_THROW_IF(output_shapes.empty(), std::invalid_argument)
    << "output_shape size empty not supported";
  NNTR_THROW_IF(output_shapes.size() > 1, std::invalid_argument)
    << "output_shape size > 1 is not supported";

  indicies = std::vector<unsigned int>(data_size_);
  std::iota(indicies.begin(), indicies.end(), 0);
  std::shuffle(indicies.begin(), indicies.end(), rng);
}

void MultiDataLoader::next(float **input, float **label, bool *last) {

  auto fill_input = [this](float *input, unsigned int length,
                           unsigned int value) {
    for (unsigned int i = 0; i < length; ++i) {
      *input = value;
      input++;
    }
  };

  auto fill_label = [this](float *input, unsigned int length,
                           unsigned int value) {
    for (unsigned int i = 0; i < length; ++i) {
      *input = value;
      input++;
    }
  };

  float **cur_input_tensor = input;
  for (unsigned int i = 0; i < input_shapes.size(); ++i) {
    fill_input(*cur_input_tensor, input_shapes.at(i).getFeatureLen(),
               indicies[count]);
    cur_input_tensor++;
  }

  float **cur_label_tensor = label;
  for (unsigned int i = 0; i < output_shapes.size(); ++i) {
    fill_label(*cur_label_tensor, output_shapes.at(i).getFeatureLen(), 1);
    cur_label_tensor++;
  }

  if (updateIteration(iteration, data_size)) {
    std::shuffle(indicies.begin(), indicies.end(), rng);
    *last = true;
    count = 0;
  } else {
    *last = false;
    count++;
  }
}

} // namespace nntrainer::util
