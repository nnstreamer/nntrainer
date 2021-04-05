// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file interpreter.h
 * @date 01 April 2021
 * @brief NNTrainer interpreter that reads and generates a graphRepresentation
 from a file
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 * @note The boundary of graph interpreter is restricted to graph only.
 * @details
 * Graph is convertible either from a file, representation by a appropriate
 interpreter
 * For example, if istream would be from a a.tflite file,
 *
 * GraphRepresentaion g;
 * GraphInterpreter * interpreter = new TfliteInterpreter;
 *
 * std::ifstream f = std::open("a.tflite");
 * g = interpreter->serialize(f);
 *
 *         +--------+
 *         |iostream|
 *         +--+--+--+
 *            ^  |
 * serialize()|  |
 *            |  |
 *        (Interpreter)
 *            |  |
 *            |  | deserialize()
 *            |  v
 *    +-------+--+--------+
 *    |GraphRepresentation|
 *    +-------+-----------+
 *
 */
#ifndef __INTERPRETER_H__
#define __INTERPRETER_H__

#include <memory>
#include <string>

#include <network_graph.h>

namespace nntrainer {

using GraphRepresentation = NetworkGraph;

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
   * @param out output file name
   */
  virtual void
  serialize(std::shared_ptr<const GraphRepresentation> representation,
            const std::string &out) = 0;

  /**
   * @brief deserialize graph from a stream
   *
   * @param in input file name
   * @return GraphRepresentation graph representation
   */
  virtual std::shared_ptr<GraphRepresentation>
  deserialize(const std::string &in) = 0;
};

} // namespace nntrainer

#endif /** __INTERPRETER_H__ */
