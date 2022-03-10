// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   lr_scheduler_exponential.cpp
 * @date   09 December 2021
 * @brief  This is Exponential Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>

#include <common_properties.h>
#include <lr_scheduler_exponential.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

ExponentialLearningRateScheduler::ExponentialLearningRateScheduler() :
  lr_props(props::DecayRate(), props::DecaySteps()) {}

void ExponentialLearningRateScheduler::finalize() {
  NNTR_THROW_IF(std::get<props::DecayRate>(lr_props).empty(),
                std::invalid_argument)
    << "[ConstantLearningRateScheduler] Decay Rate is not set";
  NNTR_THROW_IF(std::get<props::DecaySteps>(lr_props).empty(),
                std::invalid_argument)
    << "[ConstantLearningRateScheduler] Decay Steps is not set";
  ConstantLearningRateScheduler::finalize();
}

void ExponentialLearningRateScheduler::setProperty(
  const std::vector<std::string> &values) {
  auto left = loadProperties(values, lr_props);
  ConstantLearningRateScheduler::setProperty(left);
}

void ExponentialLearningRateScheduler::exportTo(
  Exporter &exporter, const ExportMethods &method) const {
  ConstantLearningRateScheduler::exportTo(exporter, method);
  exporter.saveResult(lr_props, method, this);
}

double ExponentialLearningRateScheduler::getLearningRate(size_t iteration) {
  auto const &lr = ConstantLearningRateScheduler::getLearningRate(iteration);
  auto const &[decay_rate, decay_steps] = lr_props;

  return lr * pow(decay_rate, (iteration / (float)decay_steps));
}

} // namespace nntrainer
