// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   matmul_layer.cpp
 * @date   26 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is matrix multiplication layer class (operation layer)
 */

#include "common_properties.h"
#include <matmul_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

#include <layer_context.h>

namespace nntrainer {

void MatMulLayer::finalize(InitLayerContext &context) {
  TensorDim inputDim0 = context.getInputDimensions()[0];
  TensorDim inputDim1 = context.getInputDimensions()[1];
  if (inputDim0[1] != inputDim1[1]) {
    throw std::invalid_argument("MatMulLayer requires matching channel size. ");
  } else if (inputDim0[3] != inputDim1[2]) {
    throw std::invalid_argument(
      "MatMulLayer requires matching inner dimensions. but got " +
      std::to_string(inputDim0[3]) + "!= " + std::to_string(inputDim1[2]));
  }

  TensorDim output_dim = TensorDim(inputDim0);
  output_dim.setTensorDim(3, inputDim1[3]);
  context.setOutputDimensions({std::move(output_dim)});
}

void MatMulLayer::forwarding_operation(const Tensor &input0,
                                       const Tensor &input1, Tensor &output) {

  if (input0.channel() == 16 && input0.height() == 256 &&
      input0.width() == 128) {

    Tensor newInput0 = input0;
    newInput0.reshape(TensorDim({16, 1, 256, 128}));

    Tensor newInput1 = input1;
    newInput1 = newInput1.transpose("0:2:1");
    newInput1.reshape(TensorDim({16, 1, 256, 128}));

    output.reshape(TensorDim({16, 1, 256, 256}));
    newInput0.dotBatched(newInput1, output, false, true);
    output.reshape(TensorDim({1, 16, 256, 256}));
  } else if (input0.channel() == 16 && input0.height() == 256 &&
             input0.width() == 256) {

    Tensor newInput0 = input0;
    newInput0.reshape(TensorDim({16, 1, 256, 256}));

    Tensor newInput1 = input1;
    newInput1.reshape(TensorDim({16, 1, 256, 128}));

    output.reshape(TensorDim({16, 1, 256, 128}));
    newInput0.dotBatched(newInput1, output);
    output.reshape(TensorDim({1, 16, 256, 128}));

  } else
    input0.dot(input1, output);
}

void MatMulLayer::calcDerivative(RunLayerContext &context) {
  const Tensor &inDeriv = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &outDeriv0 = context.getOutgoingDerivative(0);
  Tensor &outDeriv1 = context.getOutgoingDerivative(1);
  const Tensor &input0 = context.getInput(0);
  const Tensor &input1 = context.getInput(1);

  inDeriv.dot(input1, outDeriv0, false, true);
  input0.dot(inDeriv, outDeriv1, true, false);
}

void MatMulLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, matmul_props);
  if (!remain_props.empty()) {
    std::string msg = "[MatMulLayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

} /* namespace nntrainer */
