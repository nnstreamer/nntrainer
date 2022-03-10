// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lr_scheduler_constant.cpp
 * @date   09 December 2021
 * @brief  This is Constant Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>

#include <common_properties.h>
#include <lr_scheduler_constant.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

ConstantLearningRateScheduler::ConstantLearningRateScheduler() :
  lr_props(props::LearningRate()) {}

void ConstantLearningRateScheduler::finalize() {
  NNTR_THROW_IF(std::get<props::LearningRate>(lr_props).empty(),
                std::invalid_argument)
    << "[ConstantLearningRateScheduler] Learning Rate is not set";
}

void ConstantLearningRateScheduler::setProperty(
  const std::vector<std::string> &values) {
  auto left = loadProperties(values, lr_props);
  NNTR_THROW_IF(left.size(), std::invalid_argument)
    << "[ConstantLearningRateScheduler] There are unparsed properties";
}

void ConstantLearningRateScheduler::exportTo(
  Exporter &exporter, const ExportMethods &method) const {
  exporter.saveResult(lr_props, method, this);
}

double ConstantLearningRateScheduler::getLearningRate(size_t iteration) {
  return std::get<props::LearningRate>(lr_props);
}

} // namespace nntrainer
