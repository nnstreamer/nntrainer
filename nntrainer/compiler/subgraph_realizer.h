// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Eunju Yang <ej.yang@samsung.com>
 *
 * @file subgraph_realizer.h
 * @date 27 Dec 2024
 * @brief NNTrainer subgraph realizer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __SUBGRAPH_REALIZER_H__
#define __SUBGRAPH_REALIZER_H__

#include <realizer.h>

#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include <connection.h>

namespace nntrainer {

namespace props {
/**
 * @brief Property subgraph_idx
 */
class SubgraphIdx final : public nntrainer::Property<unsigned int> {
public:
  SubgraphIdx(const unsigned int &val = 0) :
    nntrainer::Property<unsigned int>(val) {
    set(val);
  }
  static constexpr const char *key = "subgraph_idx";
  using prop_tag = uint_prop_tag;
};

/**
 * @brief Property is_shared_subgraph
 */
class IsSharedSubgraph final : public nntrainer::Property<bool> {
public:
  IsSharedSubgraph(bool val = true) : nntrainer::Property<bool>(val) {}
  static constexpr const char *key = "is_shared_subgraph";
  using prop_tag = bool_prop_tag;
};
} // namespace props

/**
 * @brief SubGraph Realizer which adding some properties for subgraph
 * construction.
 * @param properties
 *  subgraph_idx = <int>
 *  is_shared_subgraph = <bool>
 */
class SubgraphRealizer : public GraphRealizer {
public:
  /**
   * @brief Construct a new Subgraph Realizer object
   * @note
   * SubGraphRealizer do the two tasks:
   *  1. Update name of the every node in subgraph with scope/
   *  2. The scope name can be varied according to its subgraph index, i.e.,
   * scope/idx/name
   *  3. If is_shared_subgraph is true, then the scope name will be shared among
   * subgraphs.
   * @param properties
   *   subgraph_idx = <int>
   *   is_shared_subgraph = <bool>
   * @param input_conns input conns from outer side
   */
  SubgraphRealizer(const std::string scope,
                   const std::vector<std::string> &properties,
                   const std::vector<Connection> &input_conns);

  /**
   * @brief Destroy the subgraph realizer object
   */
  ~SubgraphRealizer();

  /**
   * @brief realized graph
   */
  GraphRepresentation realize(const GraphRepresentation &reference) override;

private:
  std::tuple<props::SubgraphIdx, props::IsSharedSubgraph>
    realizer_props; /**< subgraph properties */
  std::vector<Connection> input_conns;
  std::string scope_name; /**< scope name */
  std::string scope_full_name;
  std::unordered_map<std::string, std::string>
    subgraph_node_names; /**< subgraph_name, original name */
  unsigned int subgraph_idx;
  bool is_shared_subgraph;
};

} /* namespace nntrainer */

#endif /* __SUBGRAPH_REALIZER_H__ */
