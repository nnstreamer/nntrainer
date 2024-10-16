// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   kld_loss_layer.cpp
 * @date   25 November 2021
 * @brief  KLD (Kullback-Leibler Divergence) loss implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <kld_loss_layer.h>
#include <layer_context.h>
#include <string>
#include <vector>

namespace nntrainer {
KLDLossLayer::KLDLossLayer() {}

KLDLossLayer::~KLDLossLayer() {}

void KLDLossLayer::setProperty(const std::vector<std::string> &values) {
  if (values.size()) {
    throw std::invalid_argument(
      "kld loss does not take any properties, but values given");
  }
}

void KLDLossLayer::forwarding(RunLayerContext &context, bool training) {
  // -0.5 * sum(1 + log_std - pow(mu, 2) - exp(log_std))
  auto &mu = context.getInput(0);
  auto &log_std = context.getInput(1);
  auto &ret = context.getOutput(0);
  auto &temp = context.getTensor(temp_idx);
  auto &before_sum = context.getTensor(before_sum_idx);

  mu.pow(2.0f, temp);                 // 1. temp = mu ^ 2
  log_std.subtract(temp, before_sum); // 2. before_sum = log_std - temp
  log_std.apply<float>(expf, temp);   // 3. temp = exp(log_std) - 1
  temp.subtract_i(1.0f);
  before_sum.subtract_i(temp);          // 4. before_sum = before_sum - temp
  before_sum.sum({1, 2, 3}, ret, -0.5); // 5. sum * 0.5
}

void KLDLossLayer::calcDerivative(RunLayerContext &context) {
  auto &d_incoming = context.getIncomingDerivative(0);
  auto &mu = context.getInput(0);

  auto &temp = context.getTensor(temp_idx);

  auto &d_mu = context.getOutgoingDerivative(0);
  auto &d_var = context.getOutgoingDerivative(1);

  // d_mu = d_incoming * mu
  mu.multiply(d_incoming, d_mu);

  // temp is exp(log_std) - 1;
  // d_var =  d_incoming * (-0.5) * ( 1 - exp(log_std) )
  //       =  d_incoming * (0.5) * ( temp )
  temp.multiply(d_incoming.multiply(0.5), d_var);
}

void KLDLossLayer::setBatch(nntrainer::RunLayerContext &context,
                            unsigned int batch) {
  context.updateTensor(temp_idx, batch);
  context.updateTensor(before_sum_idx, batch);
}
} // namespace nntrainer
