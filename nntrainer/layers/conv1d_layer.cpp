// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv1d_layer.h
 * @date   13 Oct 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution 1D Layer Class for Neural Network
 *
 */
#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <blas_interface.h>
#include <conv1d_layer.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

Conv1DLayer::Conv1DLayer(const std::array<unsigned int, 2> &padding_) :
  LayerImpl(),
  padding(padding_),
  conv_props(props::FilterSize(), props::KernelSize(), props::Stride(),
             props::Padding1D()),
  wt_idx({0}) {}
void Conv1DLayer::finalize(InitLayerContext &context) {}

void Conv1DLayer::forwarding(RunLayerContext &context, bool training) {}

void Conv1DLayer::calcDerivative(RunLayerContext &context) {}

void Conv1DLayer::calcGradient(RunLayerContext &context) {}

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
