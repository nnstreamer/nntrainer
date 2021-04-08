// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   layer_node.cpp
 * @date   1 April 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer node for network graph
 */

#include <layer_node.h>

namespace nntrainer {

std::shared_ptr<Layer> getLayerDevel(std::shared_ptr<ml::train::Layer> l) {
  std::shared_ptr<LayerNode> lnode = std::static_pointer_cast<LayerNode>(l);

  std::shared_ptr<Layer> &layer = lnode->getObject();

  return layer;
}

}; // namespace nntrainer
