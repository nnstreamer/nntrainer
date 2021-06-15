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

const std::string CenteringLayer::type = "centering";

CenteringLayer::CenteringLayer(const std::string &feature_path_) :
  LayerV1(),
  feature_path(feature_path_) {}

int CenteringLayer::setProperty(std::vector<std::string> values) {
  const std::string FEATURE_PATH("feature_path");
  util::Entry e;

  std::vector<std::string> unhandled_values;

  for (auto &val : values) {
    try {
      e = util::getKeyValue(val);
    } catch (std::invalid_argument &e) {
      std::cerr << e.what() << std::endl;
      return ML_ERROR_INVALID_PARAMETER;
    }

    if (e.key == FEATURE_PATH) {
      feature_path = e.value;
    } else {
      unhandled_values.push_back(val);
    }
  }

  return nntrainer::LayerV1::setProperty(unhandled_values);
}

int CenteringLayer::initialize(nntrainer::Manager &manager) {
  output_dim[0] = input_dim[0];

  return ML_ERROR_NONE;
}

void CenteringLayer::read(std::ifstream &file) {
  mean_feature_vector = nntrainer::Tensor(input_dim[0]);
  std::ifstream f(feature_path, std::ios::in | std::ios::binary);
  if (!f.good()) {
    throw std::invalid_argument(
      "[CenteringLayer::read] cannot read feature vector");
  }
  mean_feature_vector.read(f);
}

void CenteringLayer::forwarding(bool training) {
  std::cout << net_input[0]->getVariableRef().getDim();
  std::cout << net_hidden[0]->getVariableRef().getDim();
  std::cout << mean_feature_vector.getDim();
  net_input[0]->getVariableRef().add(mean_feature_vector,
                                     net_hidden[0]->getVariableRef(), -1);
}

void CenteringLayer::calcDerivative() {
  throw std::invalid_argument("[CenteringLayer::calcDerivative] This Layer "
                              "does not support backward propagation");
}

} // namespace layers
} // namespace simpleshot
