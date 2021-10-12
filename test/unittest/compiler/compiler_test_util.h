// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file compiler_test_util.h
 * @date 09 October 2021
 * @brief NNTrainer graph compiler related common functions
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __COMPILER_TEST_UTIL_H__
#define __COMPILER_TEST_UTIL_H__

#include <string>
#include <utility>
#include <vector>

#include <app_context.h>
#include <interpreter.h>

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;

/**
 * @brief make graph of a representation
 *
 * @param layer_reps layer representation (pair of type, properties)
 * @return nntrainer::GraphRepresentation synthesized graph representation
 */
nntrainer::GraphRepresentation
makeGraph(const std::vector<LayerRepresentation> &layer_reps);

/**
 * @brief resolve path for compiler test util
 * @see nntrainer::getResPath(const std::string&path) for how this works
 *
 * @param path model path
 * @return const std::string calcaulted path
 */
const std::string compilerPathResolver(const std::string &path);

/**
 * @brief prototypical version of checking graph is equal
 *
 * @param lhs compiled(later, finalized) graph to be compared
 * @param rhs compiled(later, finalized) graph to be compared
 * @retval true graph is equal
 * @retval false graph is not equal
 */
void graphEqual(const nntrainer::GraphRepresentation &lhs,
                const nntrainer::GraphRepresentation &rhs);

#endif // __COMPILER_TEST_UTIL_H__
