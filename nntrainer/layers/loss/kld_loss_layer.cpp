// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   kld_loss_layer.cpp
 * @date   25 November 2021
 * @brief  KLD (Kullback-Leibler Divergence) loss implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <kld_loss_layer.h>
#include <layer_context.h>
#include <string>
#include <vector>

namespace nntrainer {
KLDLossLayer::KLDLossLayer() {}

KLDLossLayer::~KLDLossLayer() {}

void KLDLossLayer::setProperty(const std::vector<std::string> &values) {}

void KLDLossLayer::forwarding(RunLayerContext &context, bool training) {}

void KLDLossLayer::calcDerivative(RunLayerContext &context) {}
} // namespace nntrainer
