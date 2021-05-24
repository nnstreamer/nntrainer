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

#include <layer_factory.h>
#include <layer_node.h>

namespace nntrainer {

/**
 * @brief Layer factory creator with constructor
 */
std::unique_ptr<LayerNode> createLayerNode(const std::string &type) {
  return std::make_unique<LayerNode>(createLayer(type));
}

std::ostream &operator<<(std::ostream &out, const LayerNode &l) {
  out << "[" << l.getName() << '/' << l.getType() << "]\n";
  auto print_vector = [&out](const std::vector<std::string> &layers,
                             const std::string &title) {
    out << title << "[" << layers.size() << "] ";
    for (auto &layer : layers) {
      out << layer << ' ';
    }
    out << '\n';
  };

  print_vector(l.input_layers, " input_layers");
  print_vector(l.output_layers, "output_layers");
  /// comment intended here,
  // print_vector(l.getObject()->input_layers, " input_layers");
  // print_vector(l.getObject()->output_layers, "output_layers");
  return out;
}

}; // namespace nntrainer
