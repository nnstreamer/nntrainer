// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   model_common_properties.cpp
 * @date   27 Aug 2021
 * @brief  This file contains common properties for model
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <model_common_properties.h>

#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer::props {
Epochs::Epochs(unsigned int _value) { set(_value); }

bool LossType::isValid(const std::string &_value) const {
  ml_logw("Model loss property is deprecated, use loss layer directly instead");
  return istrequal(_value, "cross") || istrequal(_value, "mse") ||
         istrequal(_value, "kld");
}

TrainingBatchSize::TrainingBatchSize(unsigned int _value) { set(_value); }

ContinueTrain::ContinueTrain(bool _value) { set(_value); }

MemoryOptimization::MemoryOptimization(bool _value) { set(_value); }

Fsu::Fsu(bool _value) { set(_value); }

FsuPath::FsuPath(const std::string &_value) { set(_value); }

FsuLookahead::FsuLookahead(const unsigned int &_value) { set(_value); }
ModelTensorDataType::ModelTensorDataType(ModelTensorDataTypeInfo::Enum _value) {
  set(_value);
}
LossScale::LossScale(float _value) { set(_value); }

bool LossScale::isValid(const float &_value) const {
  bool is_valid = (std::fpclassify(_value) != FP_ZERO);
  if (!is_valid)
    ml_loge("Loss scale cannot be 0");
  return is_valid;
}

} // namespace nntrainer::props
