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

#include <nntrainer_error.h>
#include <random>

namespace nntrainer::resnet {

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
  auto fill_last = [&last, this] {
    if (iteration++ == iteration_for_one_epoch) {
      iteration = 0;
      *last = true;
    } else {
      *last = false;
    }
    return *last;
  };

  auto fill_input = [this](float *input, unsigned int length) {
    for (unsigned int i = 0; i < length; ++i) {
      *input = input_dist(rng);
      input++;
    }
  };

  auto fill_label = [this](float *label, unsigned int batch,
                           unsigned int length) {
    for (unsigned int i = 0; i < batch; ++i) {
      unsigned int generated_label = label_dist(rng);

      switch (length) {
      case 1: { /// case of single integer value
        *label = generated_label;
        label++;
        break;
      }
      default: { /// case of one hot
        for (unsigned int j = 0; j < length; ++j) {
          *label = (generated_label == j);
          label++;
        }
        break;
      }
      }
    }
  };

  if (fill_last() == true) {
    return;
  }

  float **cur_input_tensor = input;
  for (unsigned int i = 0; i < input_shapes.size(); ++i) {
    fill_input(*cur_input_tensor, input_shapes.at(i).getDataLen());
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
                                       int splits) {
  /// NYI!
}

void Cifar100DataLoader::next(float **input, float **label, bool *last) {
  /// NYI!
  *last = true;
}

} // namespace nntrainer::resnet
