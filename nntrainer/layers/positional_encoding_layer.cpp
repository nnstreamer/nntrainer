// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   positional_encoding_layer.cpp
 * @date   16 August 2022
 * @brief  This file contains the positional encoding layer in transformer
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1607.06450
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <math.h>
#include <regex>

#include <positional_encoding_layer.h>
#include <tensor_dim.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum PositionalEncodingParams {
  positional_encoding,
};

PositionalEncodingLayer::PositionalEncodingLayer() :
  positional_encoding_props(props::MaxTimestep()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

PositionalEncodingLayer::~PositionalEncodingLayer() {}

void PositionalEncodingLayer::finalize(InitLayerContext &context) {
  unsigned int max_token_size =
    std::get<props::MaxTimestep>(positional_encoding_props);

  std::vector<ml::train::TensorDim> input_dims = context.getInputDimensions();
  context.setOutputDimensions(input_dims);

  unsigned int model_dim = input_dims[SINGLE_INOUT_IDX].width();

  ml::train::TensorDim pe_dim({max_token_size, model_dim});
  weight_idx[PositionalEncodingParams::positional_encoding] =
    context.requestTensor(pe_dim, "positional_encoding",
                          nntrainer::Tensor::Initializer::NONE, false,
                          nntrainer::TensorLifespan::MAX_LIFESPAN);
}

void PositionalEncodingLayer::forwarding(RunLayerContext &context,
                                         bool training) {
  const nntrainer::Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  nntrainer::Tensor &pe = context.getTensor(
    weight_idx[PositionalEncodingParams::positional_encoding]);

  TensorDim input_dim = input.getDim();
  TensorDim pe_partial_dim({input_dim.height(), input_dim.width()});
  nntrainer::Tensor pe_partial = pe.getSharedDataTensor(pe_partial_dim, 0);

  if (!isPEcalculated) {
    calculatePositionalEncoding(context);
  }

  input.add(pe_partial, output);
}

void PositionalEncodingLayer::calcDerivative(RunLayerContext &context) {
  const nntrainer::Tensor &incoming_derivative =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &outgoing_derivative =
    context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  outgoing_derivative.copyData(incoming_derivative);
}

void PositionalEncodingLayer::calculatePositionalEncoding(
  nntrainer::RunLayerContext &context) {
  unsigned int max_token_size =
    std::get<props::MaxTimestep>(positional_encoding_props);

  unsigned int model_dim = context.getInput(SINGLE_INOUT_IDX).getDim().width();

  nntrainer::Tensor &pe = context.getTensor(
    weight_idx[PositionalEncodingParams::positional_encoding]);

  float value;
  for (unsigned int i = 0; i < max_token_size; ++i) {
    for (unsigned int j = 0; j < model_dim; ++j) {
      unsigned int jj = (j >> 1) << 1;
      value = i / powf(10000.0f, jj / (float)model_dim);
      if (j & 1) {
        value = cosf(value);
      } else {
        value = sinf(value);
      }
      pe.setValue(0, 0, i, j, value);
    }
  }

  isPEcalculated = true;
}

void PositionalEncodingLayer::setProperty(
  const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, positional_encoding_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[positional encoding layer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void PositionalEncodingLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  exporter.saveResult(positional_encoding_props, method, this);
}

} /* namespace nntrainer */
