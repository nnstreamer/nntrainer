// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 *
 * @file   lr_scheduler_linear.cpp
 * @date   11 November 2024
 * @brief  This is Linear Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>

#include <common_properties.h>
#include <lr_scheduler_linear.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

LinearLearningRateScheduler::LinearLearningRateScheduler() :
  lr_props(props::MaxLearningRate(), props::MinLearningRate(),
           props::DecaySteps()) {}

void LinearLearningRateScheduler::finalize() {
  NNTR_THROW_IF(std::get<props::MaxLearningRate>(lr_props).empty(),
                std::invalid_argument)
    << "[LinearLearningRateScheduler] Max Learning Rate is not set";
  NNTR_THROW_IF(std::get<props::MinLearningRate>(lr_props).empty(),
                std::invalid_argument)
    << "[LinearLearningRateScheduler] Min Learning Rate is not set";
  NNTR_THROW_IF(std::get<props::DecaySteps>(lr_props).empty(),
                std::invalid_argument)
    << "[LinearLearningRateScheduler] Decay Steps is not set";
  NNTR_THROW_IF(std::get<props::DecaySteps>(lr_props) <= 0,
                std::invalid_argument)
    << "[LinearLearningRateScheduler] Decay Steps must be a positive integer";
}

void LinearLearningRateScheduler::setProperty(
  const std::vector<std::string> &values) {
  auto left = loadProperties(values, lr_props);
  NNTR_THROW_IF(left.size(), std::invalid_argument)
    << "[LinearLearningRateScheduler] There are unparsed properties";
}

void LinearLearningRateScheduler::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(lr_props, method, this);
}

double LinearLearningRateScheduler::getLearningRate(size_t iteration) {
  auto const &max_lr = std::get<props::MaxLearningRate>(lr_props);
  auto const &min_lr = std::get<props::MinLearningRate>(lr_props);
  auto const &decay_steps = std::get<props::DecaySteps>(lr_props);

  // Linear formula
  double lr = max_lr - (max_lr - min_lr) * (iteration / (double)decay_steps);

  return std::max(lr, (double)min_lr);
}

} // namespace nntrainer
