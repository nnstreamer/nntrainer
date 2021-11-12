// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   mol_attention_layer.cpp
 * @date   11 November 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MoL Attention Layer Class for Neural Network
 *
 */

#include <mol_attention_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

MoLAttentionLayer::MoLAttentionLayer() : wt_idx({0}) {}

MoLAttentionLayer::~MoLAttentionLayer() {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum AttentionParams { query = 0, value = 1, key = 2, score, weights };

void MoLAttentionLayer::finalize(InitLayerContext &context) {
  finalizeCommon(context);
  /** NYI */
}

void MoLAttentionLayer::forwarding(RunLayerContext &context, bool training) {
  /** NYI */
}

void MoLAttentionLayer::calcDerivative(RunLayerContext &context) { /** NYI */ }

void MoLAttentionLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, mol_props);
  AttentionLayer::setProperty(remain_props);
}

void MoLAttentionLayer::setBatch(RunLayerContext &context, unsigned int batch) {
  /** NYI */
}

} /* namespace nntrainer */
