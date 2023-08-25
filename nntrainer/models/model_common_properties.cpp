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
Epochs::Epochs(unsigned int value) { set(value); }

bool LossType::isValid(const std::string &value) const {
  ml_logw("Model loss property is deprecated, use loss layer directly instead");
  return istrequal(value, "cross") || istrequal(value, "mse");
}

TrainingBatchSize::TrainingBatchSize(unsigned int value) { set(value); }

ContinueTrain::ContinueTrain(bool value) { set(value); }

MemoryOptimization::MemoryOptimization(bool value) { set(value); }

MemorySwap::MemorySwap(bool value) { set(value); }

MemorySwapPath::MemorySwapPath(const std::string &value) { set(value); }

MemorySwapLookahead::MemorySwapLookahead(const unsigned int &value) {
  set(value);
}
ModelTensorDataType::ModelTensorDataType(ModelTensorDataTypeInfo::Enum value) {
  set(value);
}
} // namespace nntrainer::props
