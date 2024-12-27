// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Eunju Yang <ej.yang@samsung.com>
 *
 * @file subgraph_realizer.cpp
 * @date 27 Dec 2024
 * @brief NNTrainer subgraph realizer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <base_properties.h>
#include <common_properties.h>

#include <layer_node.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <remap_realizer.h>
#include <subgraph_realizer.h>

#include <tuple>

namespace nntrainer {

SubgraphRealizer::SubgraphRealizer(const std::string scope,
                                   const std::vector<std::string> &properties,
                                   const std::vector<Connection> &input_conns) :
  realizer_props(props::SubgraphIdx(), props::IsSharedSubgraph()),
  input_conns(input_conns),
  scope_name(scope),
  subgraph_idx(0),
  is_shared_subgraph(true) {

  auto left = loadProperties(properties, realizer_props);
  NNTR_THROW_IF(!left.empty(), std::invalid_argument)
    << "There is unparsed properties";

  subgraph_idx = std::get<props::SubgraphIdx>(realizer_props).get();
  is_shared_subgraph = std::get<props::IsSharedSubgraph>(realizer_props).get();
  scope_full_name = scope + "/" + std::to_string(subgraph_idx);
}

SubgraphRealizer::~SubgraphRealizer() {}

/**
 * @note
 * subgraphrealize conducts the following two steps:
 *  1. rename all the nodes in the subgraph
 *     to be unique by adding scope as its prefix
 *  2. set shared_from property of each node
 *     in the subgraph to point to the original node
 */
GraphRepresentation
SubgraphRealizer::realize(const GraphRepresentation &reference) {

  auto subgraph_realizer = [this](const GraphRepresentation &reference_) {
    RemapRealizer rename_mapper([this](std::string &name) {
      for (auto &i : input_conns) {
        if (i.getName() == name) {
          return;
        }
      }
      std::string scoped_name = scope_full_name + "/" + name;
      subgraph_node_names[scoped_name] = name;
      name = scoped_name;
    });

    auto nodes = rename_mapper.realize(reference_);

    if (is_shared_subgraph) {
      for (auto &node : nodes)
        node->setProperty({"shared_from=" + scope_name + "/0/" +
                           subgraph_node_names[node->getName()]});
    }
    return nodes;
  };

  return subgraph_realizer(reference);
}

} // namespace nntrainer
