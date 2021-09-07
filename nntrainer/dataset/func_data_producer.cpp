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

#include <base_properties.h>
#include <nntrainer_error.h>
#include <node_exporter.h>

namespace nntrainer {

/**
 * @brief User data props
 *
 */
class PropsUserData final : public Property<void *> {
public:
  static constexpr const char *key = "user_data";
  PropsUserData(void *user_data) { set(user_data); }
  using prop_tag = ptr_prop_tag;
};

FuncDataProducer::FuncDataProducer(datagen_cb datagen_cb, void *user_data_) :
  cb(datagen_cb),
  user_data_prop(new PropsUserData(user_data_)) {}

FuncDataProducer::~FuncDataProducer() {}

const std::string FuncDataProducer::getType() const {
  return FuncDataProducer::type;
}

void FuncDataProducer::setProperty(const std::vector<std::string> &properties) {
  auto left = loadProperties(properties, std::tie(*user_data_prop));
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "properties is not empty, size: " << properties.size();
}

DataProducer::Generator
FuncDataProducer::finalize(const std::vector<TensorDim> &input_dims,
                           const std::vector<TensorDim> &label_dims,
                           void *user_data) {
  NNTR_THROW_IF(!this->cb, std::invalid_argument)
    << "given callback is nullptr!";

  auto input_data = std::shared_ptr<float *>(new float *[input_dims.size()],
                                             std::default_delete<float *[]>());
  auto label_data = std::shared_ptr<float *>(new float *[label_dims.size()],
                                             std::default_delete<float *[]>());

  return [cb = this->cb, ud = this->user_data_prop->get(), input_data,
          label_data](unsigned int idx, std::vector<Tensor> &inputs,
                      std::vector<Tensor> &labels) -> bool {
    float **input_data_raw = input_data.get();
    float **label_data_raw = label_data.get();

    for (unsigned int i = 0; i < inputs.size(); ++i) {
      *(input_data_raw + i) = inputs[i].getData();
    }

    for (unsigned int i = 0; i < labels.size(); ++i) {
      *(label_data_raw + i) = labels[i].getData();
    }

    bool last = false;
    int status = cb(input_data_raw, label_data_raw, &last, ud);
    NNTR_THROW_IF(status != ML_ERROR_NONE, std::invalid_argument)
      << "[DataProducer] Callback returned error: " << status << '\n';

    return last;
  };
}

void FuncDataProducer::exportTo(Exporter &exporter,
                                const ExportMethods &method) const {}

} // namespace nntrainer
