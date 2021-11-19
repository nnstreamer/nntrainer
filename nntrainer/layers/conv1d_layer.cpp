// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv1d_layer.h
 * @date   13 Oct 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution 1D Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <conv1d_layer.h>
#include <conv2d_layer.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

Conv1DLayer::Conv1DLayer(const std::array<unsigned int, 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(props::FilterSize(), props::KernelSize(), props::Stride(),
             props::Padding2D()),
  wt_idx({0}) {
  conv2d_layer = std::make_unique<Conv2DLayer>();
}

Conv1DLayer::~Conv1DLayer() {}

void Conv1DLayer::finalize(InitLayerContext &context) {
  if (context.getNumInputs() != 1) {
    throw std::invalid_argument("Convolution layer takes only one input");
  }

  if (context.getInputDimensions()[SINGLE_INOUT_IDX].height() != 1) {
    throw std::invalid_argument("Conv1D layer requires input with height 1");
  }

  /** set the given properties as key value pair */
  auto setPropertyKV = [this](const std::string &key,
                              const std::string &value) {
    auto const &prop = key + "=" + value;
    conv2d_layer->setProperty({prop});
  };

  setPropertyKV(props::FilterSize::key,
                std::to_string(std::get<props::FilterSize>(conv_props).get()));
  setPropertyKV(
    props::KernelSize::key,
    "1," + std::to_string(std::get<props::KernelSize>(conv_props).get()));
  setPropertyKV(props::Stride::key,
                "1," +
                  std::to_string(std::get<props::Stride>(conv_props).get()));
  setPropertyKV(props::Padding2D::key,
                std::get<props::Padding2D>(conv_props).get());

  conv2d_layer->finalize(context);
}

void Conv1DLayer::forwarding(RunLayerContext &context, bool training) {
  conv2d_layer->forwarding(context, training);
}

void Conv1DLayer::calcDerivative(RunLayerContext &context) {
  conv2d_layer->calcDerivative(context);
}

void Conv1DLayer::calcGradient(RunLayerContext &context) {
  conv2d_layer->calcGradient(context);
}

void Conv1DLayer::exportTo(Exporter &exporter,
                           const ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(conv_props, method, this);
}

void Conv1DLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, conv_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace nntrainer
