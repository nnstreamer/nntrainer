// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   func_data_producer.cpp
 * @date   12 July 2021
 * @brief  This file contains various data producers from a callback
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <func_data_producer.h>

#include <nntrainer_error.h>

namespace nntrainer {

FuncDataProducer::FuncDataProducer(datagen_cb datagen_cb, void *user_data_) :
  cb(datagen_cb),
  user_data(user_data_) {}

FuncDataProducer::~FuncDataProducer() {}

const std::string FuncDataProducer::getType() const {
  return FuncDataProducer::type;
}

void FuncDataProducer::setProperty(const std::vector<std::string> &properties) {
  NNTR_THROW_IF(!properties.empty(), std::invalid_argument)
    << "properties is not empty, size: " << properties.size();
}

DataProducer::Gernerator
FuncDataProducer::finalize(const std::vector<TensorDim> &input_dims,
                           const std::vector<TensorDim> &label_dims) {
  NNTR_THROW_IF(!this->cb, std::invalid_argument)
    << "given callback is nullptr!";

  auto input_data = std::shared_ptr<float *>(new float *[input_dims.size()],
                                             std::default_delete<float *[]>());
  auto label_data = std::shared_ptr<float *>(new float *[label_dims.size()],
                                             std::default_delete<float *[]>());

  return [cb = this->cb, ud = this->user_data, input_dims, label_dims,
          input_data, label_data]() -> DataProducer::Iteration {
    std::vector<Tensor> inputs;
    inputs.reserve(input_dims.size());

    float **input_data_raw = input_data.get();
    float **label_data_raw = label_data.get();

    for (unsigned int i = 0; i < input_dims.size(); ++i) {
      inputs.emplace_back(input_dims[i]);
      *(input_data_raw + i) = inputs.back().getData();
    }

    std::vector<Tensor> labels;
    labels.reserve(label_dims.size());

    for (unsigned int i = 0; i < label_dims.size(); ++i) {
      labels.emplace_back(label_dims[i]);
      *(label_data_raw + i) = labels.back().getData();
    }

    bool last = false;
    int status = cb(input_data_raw, label_data_raw, &last, ud);
    NNTR_THROW_IF(status != ML_ERROR_NONE, std::invalid_argument)
      << "[DataProducer] Callback returned error: " << status << '\n';

    if (last) {
      return {true, {}, {}};
    } else {
      return {false, inputs, labels};
    }
  };
}
} // namespace nntrainer
