// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimizer_impl.cpp
 * @date   18 March 2021
 * @brief  This is base Optimizer implementation class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <fstream>
#include <iostream>

#include <cmath>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <optimizer_impl.h>
#include <util_func.h>

namespace nntrainer {

OptimizerImpl::OptimizerImpl() :
  optimizer_impl_props(PropsLR(), PropsDecayRate(), PropsDecaySteps()) {}

void OptimizerImpl::setProperty(const std::vector<std::string> &values) {
  auto left = loadProperties(values, optimizer_impl_props);
  NNTR_THROW_IF(left.size(), std::invalid_argument)
    << "[OptimizerImpl] There are unparsed properties";
}

void OptimizerImpl::exportTo(Exporter &exporter,
                             const ExportMethods &method) const {
  exporter.saveResult(optimizer_impl_props, method, this);
}

double OptimizerImpl::getLearningRate(size_t iteration) const {

  auto &[float_lr, decay_rate, decay_steps] = optimizer_impl_props;
  double ll = float_lr;

  if (!decay_steps.empty() && !decay_rate.empty()) {
    ll = ll * pow(decay_rate, (iteration / (float)decay_steps));
  }

  return ll;
}

} // namespace nntrainer
