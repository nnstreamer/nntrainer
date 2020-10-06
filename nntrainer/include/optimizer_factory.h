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

#include <adam.h>
#include <optimizer_internal.h>
#include <sgd.h>

namespace nntrainer {

/**
 * @brief Factory creator with copy constructor
 */
std::unique_ptr<Optimizer> createOptimizer(OptType type, const Optimizer &opt);

/**
 * @brief Factory creator with constructor
 */
template <typename... Args>
std::unique_ptr<Optimizer> createOptimizer(OptType type, Args... args) {
  switch (type) {
  case OptType::SGD:
    return std::make_unique<SGD>(args...);
  case OptType::ADAM:
    return std::make_unique<Adam>(args...);
  case OptType::UNKNOWN:
    /** fallthrough intended */
  default:
    throw std::invalid_argument("Unknown type for the optimizer");
  }
}

} // namespace nntrainer

#endif // __cplusplus
#endif // __OPTIMIZER_FACTORY_H__
