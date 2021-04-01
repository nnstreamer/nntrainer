// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler.h
 * @date 01 April 2021
 * @brief NNTrainer compiler that reads and generates executable graph
 * @details
 * Graph is convertible either to iostream, representation, executable by
 * appropriate compiler and interpreter
 *         +--------+
 *         |iostream|
 *         +--+-----+
 *            |  ^
 * operator<< |  |
 *            |  |
 *        (Interpreter)
 *            |  |
 *            |  | operator>>
 *            |  |
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
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <iostream>
#include <memory>

namespace nntrainer {

class ExecutableGraph;

class GraphRepresentation;

/**
 * @brief Pure virtual class for the Graph Compiler
 *
 */
class GraphCompiler {
public:
  virtual ~GraphCompiler() {}
  /**
   * @brief serialize graph to a file stream
   *
   * @param representation graph representation
   * @param file ifstream to serialize graph
   */
  virtual std::shared_ptr<ExecutableGraph>
  compile(const GraphRepresentation &representation) = 0;

  /**
   * @brief deserialize graph from a file stream
   *
   * @param executable executable graph
   * @return GraphRepresentation graph representation
   */
  virtual GraphRepresentation
  decompile(std::shared_ptr<ExecutableGraph> executable) = 0;
};

/**
 * @brief Pure virtual class for the Graph Interpreter
 *
 */
class GraphInterpreter {
public:
  virtual ~GraphInterpreter() {}
  /**
   * @brief serialize graph to a stream
   *
   * @param representation graph representation
   * @param out outstream to serialize graph
   */
  virtual void serialize(const GraphRepresentation &representation,
                         std::ostream &out) = 0;

  /**
   * @brief deserialize graph from a stream
   *
   * @param in in stream to deserialize
   * @return GraphRepresentation graph representation
   */
  virtual GraphRepresentation deserialize(std::istream &in) = 0;
};

class GraphInterpreter;

/**
 * @brief a simple scaffoliding function
 *
 */
void hello_world();

} // namespace nntrainer
