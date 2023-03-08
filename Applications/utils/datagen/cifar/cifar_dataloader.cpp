// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   cifar_dataloader.h
 * @date   24 Jun 2021s
 * @brief  dataloader for cifar
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "cifar_dataloader.h"

#include <cstring>
#include <iostream>
#include <nntrainer_error.h>
#include <random>

namespace nntrainer::util {

namespace {

/**
 * @brief fill label to the given memory
 *
 * @param data data to fill
 * @param length size of the data
 * @param label label
 */
void fillLabel(float *data, unsigned int length, unsigned int label) {
  if (length == 1) {
    *data = label;
    return;
  }

  memset(data, 0, length * sizeof(float));
  *(data + label) = 1;
}

/**
 * @brief fill last to the given memory
 * @note this function increases iteration value, if last is set to true,
 * iteration resets to 0
 *
 * @param[in/out] iteration current iteration
 * @param iteration_per_epoch iteration per epoch
 * @return bool true if iteration has finished
 */
bool updateIteration(unsigned int &iteration,
                     unsigned int iteration_per_epoch) {
  if (iteration++ == iteration_per_epoch) {
    iteration = 0;
    return true;
  }
  return false;
};

} // namespace

RandomDataLoader::RandomDataLoader(const std::vector<TensorDim> &input_shapes,
                                   const std::vector<TensorDim> &output_shapes,
                                   int data_size) :
  iteration(0),
  iteration_for_one_epoch(data_size),
  input_shapes(input_shapes),
  output_shapes(output_shapes),
  input_dist(0, 255),
  label_dist(0, output_shapes.front().width() - 1) {
  NNTR_THROW_IF(output_shapes.empty(), std::invalid_argument)
    << "output_shape size empty not supported";
  NNTR_THROW_IF(output_shapes.size() > 1, std::invalid_argument)
    << "output_shape size > 1 is not supported";

  iteration_for_one_epoch /= output_shapes.front().batch();
}

void RandomDataLoader::next(float **input, float **label, bool *last) {
  auto fill_input = [this](float *input, unsigned int length) {
    for (unsigned int i = 0; i < length; ++i) {
      *input = input_dist(rng);
      input++;
    }
  };

  auto fill_label = [this](float *label, unsigned int batch,
                           unsigned int length) {
    unsigned int generated_label = label_dist(rng);
    fillLabel(label, length, generated_label);
    label += length;
  };

  if (updateIteration(iteration, iteration_for_one_epoch)) {
    *last = true;
    return;
  }

  float **cur_input_tensor = input;
  for (unsigned int i = 0; i < input_shapes.size(); ++i) {
    fill_input(*cur_input_tensor, input_shapes.at(i).getFeatureLen());
    cur_input_tensor++;
  }

  float **cur_label_tensor = label;
  for (unsigned int i = 0; i < output_shapes.size(); ++i) {
    fill_label(*label, output_shapes.at(i).batch(),
               output_shapes.at(i).getFeatureLen());
    cur_label_tensor++;
  }
}

Cifar100DataLoader::Cifar100DataLoader(const std::string &path, int batch_size,
                                       int splits) :
  batch(batch_size),
  current_iteration(0),
  file(path, std::ios::binary | std::ios::ate) {
  constexpr char error_msg[] = "failed to create dataloader, reason: ";

  NNTR_THROW_IF(!file.good(), std::invalid_argument)
    << error_msg << " Cannot open file";

  auto pos = file.tellg();
  NNTR_THROW_IF((pos % Cifar100DataLoader::SampleSize != 0),
                std::invalid_argument)
    << error_msg << " Given file does not align with the format";

  auto data_size = pos / (Cifar100DataLoader::SampleSize * splits);
  idxes = std::vector<unsigned int>(data_size);
  std::cout << "path: " << path << '\n';
  std::cout << "data_size: " << data_size << '\n';
  std::iota(idxes.begin(), idxes.end(), 0);
  std::shuffle(idxes.begin(), idxes.end(), rng);

  /// @note this truncates the remaining data of less than the batch size
  iteration_per_epoch = data_size;
}

void Cifar100DataLoader::next(float **input, float **label, bool *last) {
  /// @note below logic assumes a single input and the fine label is used

  auto fill_one_sample = [this](float *input_, float *label_, int index) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    NNTR_THROW_IF(!file.good(), std::invalid_argument)
      << "file is not good, reason: "
      << strerror_r(errno, error_buf, error_buflen);
    file.seekg(index * Cifar100DataLoader::SampleSize, std::ios_base::beg);

    uint8_t current_label;
    file.read(reinterpret_cast<char *>(&current_label), sizeof(uint8_t));
    fillLabel(label_, Cifar100DataLoader::NumClass, current_label);

    for (unsigned int i = 0; i < Cifar100DataLoader::ImageSize; ++i) {
      uint8_t data;
      file.read(reinterpret_cast<char *>(&data), sizeof(uint8_t));
      *input_ = data / 255.f;
      input_++;
    }
  };

  fill_one_sample(*input, *label, idxes[current_iteration]);
  current_iteration++;
  if (current_iteration < iteration_per_epoch) {
    *last = false;
  } else {
    *last = true;
    current_iteration = 0;
    std::shuffle(idxes.begin(), idxes.end(), rng);
  }
}

} // namespace nntrainer::util
