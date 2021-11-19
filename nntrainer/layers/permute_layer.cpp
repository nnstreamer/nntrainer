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

#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <permute_layer.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

bool props::PermuteDims::isValid(const unsigned int &value) const {
  return 0 < value && value <= 3;
}

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

void PermuteLayer::finalize(InitLayerContext &context) {
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

  initiate_direction();
  context.setOutputDimensions(
    {context.getInputDimensions()[SINGLE_INOUT_IDX].transpose(direction_str)});
}

void PermuteLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  input_.transpose(direction_str, hidden_);
}

void PermuteLayer::calcDerivative(RunLayerContext &context) {
  Tensor &hidden_grad = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_grad = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  hidden_grad.transpose(rdirection_str, input_grad);
}

void PermuteLayer::exportTo(Exporter &exporter,
                            const ExportMethods &method) const {
  exporter.saveResult(std::forward_as_tuple(direction), method);
}

void PermuteLayer::setProperty(const std::vector<std::string> &values) {
  auto left_values = loadProperties(values, std::forward_as_tuple(direction));
  if (!left_values.empty()) {
    std::string msg = "[PermuteLayer] Unknown properties set with count" +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} // namespace nntrainer
