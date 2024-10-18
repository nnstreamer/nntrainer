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
static constexpr size_t SINGLE_INOUT_IDX = 0;

KLDLossLayer::KLDLossLayer() {}

KLDLossLayer::~KLDLossLayer() {}

void KLDLossLayer::setProperty(const std::vector<std::string> &values) {
  if (values.size()) {
    throw std::invalid_argument(
      "kld loss does not take any properties, but values given");
  }
}

void KLDLossLayer::forwarding(RunLayerContext &context, bool training) {
  // Result = (P * (P / Q).log()).sum()
  // KL(P ∣∣ Q) whereP denotes the distribution of the observations in datasets
  // and Q denotes the model output.

  nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  if (context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX);
    nntrainer::Tensor temp; // temp output
    /**
     * 1. Output = label / predicted
     * 2. Output = output * label
     * 3. Output = log(output)
     * 4. Output = sum(output)
     */
    label.divide(predicted, temp);
    temp.multiply_i(label);
    temp.apply<float>(logf, temp);
    output.fill(temp.sum({0, 1, 2, 3}));
  }
}

void KLDLossLayer::calcDerivative(RunLayerContext &context) {
  /**
   * d/dQ = -P**2/(Q**2) * ln(P/Q)
   */
  nntrainer::Tensor &predicted = context.getInput(SINGLE_INOUT_IDX); // Q
  nntrainer::Tensor &label = context.getLabel(SINGLE_INOUT_IDX);     // P
  nntrainer::Tensor &deriv = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  nntrainer::Tensor temp;
  temp = label.divide(predicted);
  temp.apply<float>(logf, temp);

  label.pow_i(2);
  predicted.pow_i(2);
  label.divide_i(predicted);

  temp.multiply_i(label);
  temp.multiply_i(-1.0f);
  deriv.fill(temp);
}

} // namespace nntrainer
