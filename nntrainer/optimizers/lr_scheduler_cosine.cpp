// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 *
 * @file   lr_scheduler_cosine.cpp
 * @date   7 October 2024
 * @brief  This is CosineAnnealing Learning Rate Scheduler class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyunwoo LEE <dlgusdn0414@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>

#include <common_properties.h>
#include <lr_scheduler_cosine.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

CosineAnnealingLearningRateScheduler::CosineAnnealingLearningRateScheduler() :
  lr_props(props::MaxLearningRate(), props::MinLearningRate(),
           props::DecaySteps()) {}

void CosineAnnealingLearningRateScheduler::finalize() {
  NNTR_THROW_IF(std::get<props::MaxLearningRate>(lr_props).empty(),
                std::invalid_argument)
    << "[CosineAnnealingLearningRateScheduler] Max Learning Rate is not set";
  NNTR_THROW_IF(std::get<props::MinLearningRate>(lr_props).empty(),
                std::invalid_argument)
    << "[CosineAnnealingLearningRateScheduler] Min Learning Rate is not set";
  NNTR_THROW_IF(std::get<props::DecaySteps>(lr_props).empty(),
                std::invalid_argument)
    << "[CosineAnnealingLearningRateScheduler] Decay Steps is not set";
  CosineAnnealingLearningRateScheduler::finalize();
}

void CosineAnnealingLearningRateScheduler::setProperty(
  const std::vector<std::string> &values) {
  auto left = loadProperties(values, lr_props);
  CosineAnnealingLearningRateScheduler::setProperty(left);
}

void CosineAnnealingLearningRateScheduler::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  CosineAnnealingLearningRateScheduler::exportTo(exporter, method);
  exporter.saveResult(lr_props, method, this);
}

double CosineAnnealingLearningRateScheduler::getLearningRate(size_t iteration) {
  auto const &max_lr = std::get<props::MaxLearningRate>(lr_props);
  auto const &min_lr = std::get<props::MinLearningRate>(lr_props);
  auto const &decay_steps = std::get<props::DecaySteps>(lr_props);

  auto lr = CosineAnnealingLearningRateScheduler::getLearningRate(iteration);

  // Cosine annealing formula
  double cosine_decay =
    0.5 * (1 + cos(M_PI * (iteration % decay_steps) / (double)decay_steps));
  return min_lr + (max_lr - min_lr) * cosine_decay;
}

} // namespace nntrainer
