// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   permute_layer.cpp
 * @date   06 May 2021
 * @brief  Permute layer to support transpose
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <sstream>
#include <string>
#include <tuple>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <permute_layer.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace nntrainer {

bool props::PermuteDims::isValid(const unsigned int &value) const {
  return 0 < value && value <= 3;
}

const std::string PermuteLayer::type = "permute";

/**
 * @brief buildTransposeString based on array
 * @todo deprecate this
 *
 * @param arr array to make a representation
 * @return const std::string string to return
 */
static std::string
buildTrasposeString(const std::array<props::PermuteDims, 3> &arr) {
  std::stringstream ss;
  ss << arr[0].get() - 1 << ':' << arr[1].get() - 1 << ':' << arr[2].get() - 1;
  return ss.str();
}

int PermuteLayer::initialize(Manager &manager) {
  auto initiate_direction = [this] {
    std::bitset<3> check_transpose; /**< check if transpose contains all axis */

    for (int i = 0; i < 3; ++i) {
      check_transpose.set(direction[i] - 1, true);
      this->reverse_direction[direction[i] - 1].set(i + 1);
    }

    NNTR_THROW_IF(check_transpose.all() == false, std::invalid_argument)
      << "[Permute] "
      << "transpose direction is invalid, checked direction: "
      << check_transpose.to_string();

    /*** @todo deprecate this */
    direction_str = buildTrasposeString(direction);
    rdirection_str = buildTrasposeString(direction);
  };

  auto initiate_dimension = [this] {
    output_dim[0] = input_dim[0].transpose(direction_str);
  };

  try {
    initiate_direction();
    initiate_dimension();
  } catch (std::exception &e) {
    ml_loge("[Permute] Initiation failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

void PermuteLayer::forwarding(bool training) {
  auto &input_ = net_input[0]->getVariableRef();
  auto &hidden_ = net_hidden[0]->getVariableRef();

  input_.transpose(direction_str, hidden_);
}

void PermuteLayer::calcDerivative() {
  auto &input_grad = net_input[0]->getGradientRef();
  auto &hidden_grad = net_hidden[0]->getGradientRef();

  hidden_grad.transpose(rdirection_str, input_grad);
}

void PermuteLayer::copy(std::shared_ptr<LayerV1> l) {
  LayerV1::copy(l);

  std::shared_ptr<PermuteLayer> from =
    std::static_pointer_cast<PermuteLayer>(l);

  direction = from->direction;
  direction_str = from->direction_str;
  reverse_direction = from->reverse_direction;
  rdirection_str = from->rdirection_str;
}

void PermuteLayer::export_to(Exporter &exporter, ExportMethods method) const {
  LayerV1::export_to(exporter, method);
  exporter.saveResult(std::forward_as_tuple(direction), method);
}

int PermuteLayer::setProperty(std::vector<std::string> values) {
  try {
    auto left_values = loadProperties(values, std::forward_as_tuple(direction));
    LayerV1::setProperty(left_values);
  } catch (std::invalid_argument &e) {
    ml_loge("[PermuteLayer] failed to set property, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

} // namespace nntrainer
