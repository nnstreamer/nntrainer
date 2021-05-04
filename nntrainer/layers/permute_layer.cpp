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
#include <tuple>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <permute_layer.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace nntrainer {

const std::string PermuteLayer::type = "permute";

int PermuteLayer::initialize(Manager &manager) { /** NYI */
  return ML_ERROR_NONE;
}

void PermuteLayer::forwarding(bool training) { /** NYI */
}

void PermuteLayer::calcDerivative() { /** NYI */
}

void PermuteLayer::copy(std::shared_ptr<Layer> l) { /** NYI */
}

void PermuteLayer::export_to(Exporter &exporter, ExportMethods method) const {}

int PermuteLayer::setProperty(std::vector<std::string> values) {
  return ML_ERROR_NONE;
}

} // namespace nntrainer
