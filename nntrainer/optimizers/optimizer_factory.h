// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	optimizer_factory.h
 * @date	7 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is the optimizer factory.
 */

#ifndef __OPTIMIZER_FACTORY_H__
#define __OPTIMIZER_FACTORY_H__
#ifdef __cplusplus

#include <optimizer.h>
#include <optimizer_internal.h>

namespace nntrainer {

using OptType = ml::train::OptimizerType;

/**
 * @brief change Optimizer Type to string
 *
 * @param type type to change
 * @return const std::string string representation of the type
 */
const std::string optimizerIntToStrType(const OptType &type);
/**
 * @brief Factory creator with copy constructor
 */
std::unique_ptr<Optimizer> createOptimizer(const std::string &type,
                                           const Optimizer &opt);

/**
 * @brief Factory creator with copy constructor using enum(integer)
 */
std::unique_ptr<Optimizer> createOptimizer(const OptType &type,
                                           const Optimizer &opt);

/**
 * @brief Factory creator with constructor
 */
std::unique_ptr<Optimizer> createOptimizer(const std::string &type);

/**
 * @brief Factory creator with constructor using enum(integer)
 */
std::unique_ptr<Optimizer> createOptimizer(const OptType &type);

} // namespace nntrainer

#endif // __cplusplus
#endif // __OPTIMIZER_FACTORY_H__
