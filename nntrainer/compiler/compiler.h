// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler.h
 * @date 01 April 2021
 * @brief NNTrainer compiler that reads and generates executable graph
 * @details
 * Graph is convertible either to iostream, representation, executable by
 * appropriate compiler and interpreter.
 * For example, if istream would be from a a.tflite file,
 *
 * GraphRepresentaion g;
 * g.setInterpreter(tfliteInterpreter);
 * g.setCompiler(nntrainerCPUCompiler);
 * auto tflite_file = std::open('a.tflite');
 * tflite_file >> g;
 *
 * ExecutableGraph eg = g.compile();
 *
 * g.setInterpreter(iniInterpreter);
 * auto ini_file = std::open('a.ini');
 * ini_file << g;
 *
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
#ifndef __COMPILER_H__
#define __COMPILER_H__

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

/**
 * @brief Pure Virtual class for ExecutableGraph
 *
 * @note this class is an interface to executable graph as actual graph would be
 * different framework by framework
 *
 */
class ExecutableGraph {
  /// NYI!
};

/**
 * @brief Graph Representation
 *
 * @note this class must have the graph representation of the most expanded
 * version as far as nntrainer supports.
 *
 * GraphRepresentation contains the operation configurations, connections and
 * buffer(this might not...)
 *
 */
class GraphRepresentation {
public:
  /**
   * @brief Construct a new Graph Representation object
   *
   * @param interpreter_ interpreter to use
   * @param compiler_ compiler to use
   */
  GraphRepresentation(std::shared_ptr<GraphInterpreter> interpreter_ = nullptr,
                      std::shared_ptr<GraphCompiler> compiler_ = nullptr) :
    interpreter(interpreter_),
    compiler(compiler_) {}

  /**
   * @brief Construct a new Graph Representation object from ifstream
   *
   * @param interpreter_ interpreter to use
   * @param istream istream to convert to the representation
   */
  GraphRepresentation(std::shared_ptr<GraphInterpreter> interpreter_,
                      std::istream &in) {
    /** NYI! */
  }

  /**
   * @brief Construct a new Graph Representation object from a executable graph
   *
   * @param compiler_ compiler to use
   * @param graph graph that will be decompiled (not owing)
   */
  GraphRepresentation(std::shared_ptr<GraphCompiler> compiler_,
                      std::shared_ptr<ExecutableGraph> graph) {
    /** NYI! */
  }

  /**
   * @brief compile a graph to a executable graph
   *
   * @return std::shared_ptr<ExecutableGraph>
   */
  std::shared_ptr<ExecutableGraph> compile() {
    /** NYI !*/
    return nullptr;
  };

  /**
   * @brief decompile the graph to a representation
   *
   * @param graph graph to decompile
   */
  void decompile(std::shared_ptr<ExecutableGraph> graph){/** NYI! */};

  /**
   * @brief out operator to a file
   *
   * @param out outstream
   * @return std::ostream& outstream
   */
  std::ostream &operator<<(std::ostream &out) {
    interpreter->serialize(*this, out);
    return out;
  }

  /**
   * @brief in stream
   *
   * @param input
   * @return std::istream&
   */
  std::istream &operator>>(std::istream &input) {
    interpreter->deserialize(input);
    return input;
  }

  /**
   * @brief Set the Graph Interpreter object
   *
   * @param interpreter_ interpreter
   */
  void setGraphInterpreter(std::shared_ptr<GraphInterpreter> interpreter_) {
    interpreter = interpreter_;
  }

  /**
   * @brief Set the Graph Compiler object
   *
   * @param compiler_ compiler
   */
  void setGraphCompiler(std::shared_ptr<GraphCompiler> compiler_) {
    compiler = compiler_;
  }

private:
  std::shared_ptr<GraphInterpreter> interpreter;
  std::shared_ptr<GraphCompiler> compiler;
};

} // namespace nntrainer

#endif // __COMPILER_H__
