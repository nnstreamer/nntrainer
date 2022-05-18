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
 * @details given the iteration range [it_0, it_1, ... it_n], learning rates
 * [lr_0, lr_1 ... lr_n_1], and iteration iter, find index i such that
 * it_i_1 < iter <= it_i and return lr_i_1.
 */

#include <cmath>

#include <common_properties.h>
#include <lr_scheduler_step.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

StepLearningRateScheduler::StepLearningRateScheduler() : lr_props({}, {}) {}

void StepLearningRateScheduler::finalize() {
  NNTR_THROW_IF(std::get<std::vector<props::LearningRate>>(lr_props).empty(),
                std::invalid_argument)
    << "[StepLearningRateScheduler] Learning rates are not set";
  NNTR_THROW_IF(std::get<std::vector<props::Iteration>>(lr_props).empty(),
                std::invalid_argument)
    << "[StepLearningRateScheduler] Step iterations are not set";

  auto const &[learning_rates, iterations] = lr_props;
  if (learning_rates.size() != iterations.size() + 1)
    throw std::invalid_argument("[StepLearningRateScheduler] learning rates "
                                "and step iterations count mismatch");

  /**
   * provided step iterations must contain monotonically increasing values,
   * and all values must be unique
   */
  std::vector<unsigned int> sorted_iterations(iterations.size());
  std::transform(iterations.begin(), iterations.end(),
                 sorted_iterations.begin(),
                 [](auto const &val) { return val.get(); });
  std::sort(sorted_iterations.begin(), sorted_iterations.end());
  auto iter = std::unique(sorted_iterations.begin(), sorted_iterations.end());
  NNTR_THROW_IF(iter != sorted_iterations.end(), std::invalid_argument)
    << "[StepLearningRateScheduler] step iterations should contain unique "
       "values";
  for (unsigned int idx = 0; idx < sorted_iterations.size(); idx++)
    NNTR_THROW_IF(iterations[idx].get() != sorted_iterations[idx],
                  std::invalid_argument)
      << "[StepLearningRateScheduler] step iterations should be in "
         "monotonically increasing order";
}

void StepLearningRateScheduler::setProperty(
  const std::vector<std::string> &values) {
  auto left = loadProperties(values, lr_props);
  NNTR_THROW_IF(left.size(), std::invalid_argument)
    << "[StepLearningRateScheduler] There are unparsed properties";
}

void StepLearningRateScheduler::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(lr_props, method, this);
}

double StepLearningRateScheduler::getLearningRate(size_t iteration) {
  auto const &[learning_rates, iterations] = lr_props;
  auto upper =
    std::lower_bound(iterations.begin(), iterations.end(), iteration);
  if (upper != iterations.end())
    return learning_rates[upper - iterations.begin()];
  else
    return learning_rates.back().get();
}

} // namespace nntrainer
