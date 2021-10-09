// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler_test_util.cpp
 * @date 09 October 2021
 * @brief NNTrainer graph compiler related common functions
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <compiler_test_util.h>

#include <app_context.h>
#include <nntrainer_test_util.h>

static auto &ac = nntrainer::AppContext::Global();

const std::string compilerPathResolver(const std::string &path) {
  return getResPath(path, {"test", "test_models", "models"});
}

nntrainer::GraphRepresentation
makeGraph(const std::vector<LayerRepresentation> &layer_reps) {
  nntrainer::GraphRepresentation graph_rep;

  for (const auto &layer_representation : layer_reps) {
    /// @todo Use unique_ptr here
    std::shared_ptr<nntrainer::LayerNode> layer = createLayerNode(
      ac.createObject<nntrainer::Layer>(layer_representation.first),
      layer_representation.second);
    graph_rep.push_back(layer);
  }

  return graph_rep;
}
