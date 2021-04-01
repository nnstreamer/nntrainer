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

namespace nntrainer {

class ExecutableGraph;

class GraphRepresentation;

class GraphCompiler;

class GraphInterpreter;

/**
 * @brief a simple scaffoliding function
 *
 */
void hello_world();

} // namespace nntrainer
