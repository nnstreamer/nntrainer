// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @brief  task runner for the resnet
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << "key=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           const std::vector<T> &value) {
  if (value.empty()) {
    throw std::invalid_argument("empty vector cannot be converted");
  }

  std::stringstream ss;
  ss << "key=";
  for (unsigned int i = 0; i < value.size() - 1; ++i) {
    ss << value.at(i) << ',';
  }
  ss << value.back();

  return ss.str();
}

/**
 * @brief resnet block
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param filters number of filters
 * @param kernel_size number of kernel_size
 * @param downsample downsample to make output size 0
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> resnetBlock(const std::string &block_name,
                                     const std::string &input_name, int filters,
                                     int kernel_size, bool downsample) {
  return {};
}

/**s
 * @brief Create resnet 18
 *
 * @return ModelHandle to create model
 */
ModelHandle createResnet18() {
  std::vector<LayerHandle> layers;
  return nullptr;
}

ml_train_datagen_cb train_cb, valid_cb;

int main() {
  std::cout << "Hello world\n";

  return 0;
}
