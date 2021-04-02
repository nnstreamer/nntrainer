// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler.h
 * @date 01 April 2021
 * @brief NNTrainer compiler that reads to generate optimized graph
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 * @details
 * Graph is convertible either to iostream, representation, executable by
 * appropriate compiler and interpreter
 * For example, if istream would be from a a.tflite file,
 *
 * GraphRepresentaion g;
 * GraphCompiler * compiler = new NNTrainerCPUCompiler;
 *
 * ExecutableGraph eg = compiler->compile(g);
 *
 *
 *    +-------+--+--------+
 *    |GraphRepresentation|
 *    +-------+-----------+
 *            |  ^
 *  compile() |  |
 *            |  |
 *         (Compiler)
 *            |  |
 *            |  | decompile()
 *            v  |
 *      +--------+------+
 *      |ExecutableGraph|
 *      +---------------+
 *
 */
#ifndef __COMPILER_H__
#define __COMPILER_H__

#include <memory>

#include <network_graph.h>

namespace nntrainer {

using GraphRepresentation = NetworkGraph;
using ExecutableGraph = NetworkGraph;

/**
 * @brief Pure virtual class for the Graph Compiler
 *
 */
class GraphCompiler {
public:
  virtual ~GraphCompiler() {}
  /**
   * @brief serialize graph to a file stream
   * @todo consider adding delegates argument here when implementing it for
   * real.
   *
   * @param representation graph representation
   * @param file ifstream to serialize graph
   */
  virtual std::shared_ptr<ExecutableGraph>
  compile(std::shared_ptr<const GraphRepresentation> representation) = 0;

  /**
   * @brief deserialize graph from a file stream
   *
   * @param executable executable graph
   * @return GraphRepresentation graph representation
   */
  virtual std::shared_ptr<GraphRepresentation>
  decompile(std::shared_ptr<ExecutableGraph> executable) = 0;
};

} // namespace nntrainer

#endif // __COMPILER_H__
