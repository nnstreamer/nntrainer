// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   identity.cpp
 * @date   16 Dec 2021
 * @brief  This is identity layer flows everything as it is
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <identity_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <stdexcept>
#include <tensor.h>

namespace nntrainer {
IdentityLayer::IdentityLayer() {}

IdentityLayer::~IdentityLayer() {}

void IdentityLayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());
}

void IdentityLayer::forwarding(RunLayerContext &context, bool training) {
  if (!context.executeInPlace()) {
    for (unsigned int i = 0, sz = context.getNumInputs(); i < sz; ++i) {
      Tensor &hidden_ = context.getOutput(i);
      Tensor &input_ = context.getInput(i);
      hidden_.copyData(input_);
    }
  }
}

void IdentityLayer::calcDerivative(RunLayerContext &context) {
  if (!context.executeInPlace()) {
    for (unsigned int i = 0, sz = context.getNumInputs(); i < sz; ++i) {
      Tensor &d_hidden = context.getIncomingDerivative(i);
      Tensor &d_input = context.getOutgoingDerivative(i);
      d_input.copyData(d_hidden);
    }
  }
}

void IdentityLayer::setProperty(const std::vector<std::string> &values) {
  NNTR_THROW_IF(values.size(), std::invalid_argument)
    << "Identity layer has left unparsed properties";
}
} // namespace nntrainer
