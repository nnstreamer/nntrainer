// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   centering.cpp
 * @date   08 Jan 2021
 * @brief  This file contains the simple centering layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

#include <centering.h>
#include <simpleshot_utils.h>

namespace simpleshot {
namespace layers {

static constexpr size_t SINGLE_INOUT_IDX = 0;

CenteringLayer::CenteringLayer(const std::string &feature_path_) :
  Layer(),
  feature_path(feature_path_) {}

void CenteringLayer::setProperty(const std::vector<std::string> &values) {
  const std::string FEATURE_PATH("feature_path");
  util::Entry e;

  for (auto &val : values) {
    e = util::getKeyValue(val);

    if (e.key == FEATURE_PATH) {
      feature_path = e.value;
    } else {
      std::string msg =
        "[CenteringLayer] Unknown Layer Properties count " + val;
      throw nntrainer::exception::not_supported(msg);
    }
  }
}

void CenteringLayer::finalize(nntrainer::InitLayerContext &context) {
  context.setOutputDimensions(context.getInputDimensions());

  /** TODO: update this to requestTensor once it support init with file */
  const auto &input_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  mean_feature_vector = nntrainer::Tensor(input_dim);
  std::ifstream f(feature_path, std::ios::in | std::ios::binary);
  if (!f.good()) {
    throw std::invalid_argument(
      "[CenteringLayer::read] cannot read feature vector");
  }
  mean_feature_vector.read(f);
}

void CenteringLayer::forwarding(nntrainer::RunLayerContext &context,
                                bool training) {
  auto &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  auto &input_ = context.getInput(SINGLE_INOUT_IDX);

  std::cout << input_.getDim();
  std::cout << hidden_.getDim();
  std::cout << mean_feature_vector.getDim();
  input_.add(mean_feature_vector, hidden_, -1);
}

void CenteringLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  throw std::invalid_argument("[CenteringLayer::calcDerivative] This Layer "
                              "does not support backward propagation");
}

} // namespace layers
} // namespace simpleshot
