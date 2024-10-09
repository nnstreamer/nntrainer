// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   transpose_cl.cpp
 * @date   31 July 2024
 * @brief  Implementation of transpose layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "transpose_cl.h"
#include <blas_kernel_interface.h>
#include <iostream>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void TransposeLayerCl::finalize(InitLayerContext &context) {
  std::vector<TensorDim> dim = context.getInputDimensions();

  for (unsigned int i = 0; i < dim.size(); ++i) {
    if (dim[i].getDataLen() == 0) {
      throw std::invalid_argument("Input dimension is not set");
    }
  }

  context.setOutputDimensions(dim);
}

void TransposeLayerCl::forwarding(RunLayerContext &context, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  // "1:0:2" is arbitrary
  transposeCl("1:0:2", in, out);
}

void TransposeLayerCl::incremental_forwarding(RunLayerContext &context,
                                              unsigned int from,
                                              unsigned int to, bool training) {
  Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }
  // "1:0:2" is arbitrary
  transposeCl("1:0:2", in, out);
}

void TransposeLayerCl::calcDerivative(RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

void TransposeLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, transpose_props);
  if (!remain_props.empty()) {
    std::string msg = "[TransposeLayerCl] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

#ifdef PLUGGABLE

Layer *create_transpose_layer_cl() {
  auto layer = new TransposeLayerCl();
  return layer;
}

void destroy_transpose_layer_cl(Layer *layer) { delete layer; }

extern "C" {
LayerPluggable ml_train_layer_pluggable{create_transpose_layer_cl,
                                        destroy_transpose_layer_cl};
}

#endif

} // namespace nntrainer
