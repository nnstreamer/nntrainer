// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   layer_impl.h
 * @date   21 June 2021
 * @brief  This is base Optimizer implementation class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <layer_impl.h>

#include <string>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>

namespace nntrainer {

LayerImpl::LayerImpl() :
  finalized(false),
  layer_impl_props(std::make_unique<std::tuple<>>()) {}

void LayerImpl::finalize(InitContext &context) {
  NNTR_THROW_IF(finalized, nntrainer::exception::not_supported)
    << "[LayerImpl] "
    << "it is prohibited to finalize a layer twice";
  finalized = true;
}

void LayerImpl::setProperty(const std::vector<std::string> &values) {}

void LayerImpl::exportTo(Exporter &exporter,
                         const ExportMethods &method) const {}

} // namespace nntrainer
