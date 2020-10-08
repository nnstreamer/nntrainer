// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	optimizer_factory.cpp
 * @date	7 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is the optimizer factory.
 */

#include <adam.h>
#include <optimizer.h>
#include <sgd.h>

namespace nntrainer {

/**
 * @brief Factory creator with copy constructor
 */
std::unique_ptr<Optimizer> createOptimizer(OptType type, const Optimizer &opt) {
  switch (type) {
  case OptType::sgd:
    return std::make_unique<SGD>(static_cast<const SGD &>(opt));
  case OptType::adam:
    return std::make_unique<Adam>(static_cast<const Adam &>(opt));
  case OptType::unknown:
    /** fallthrough intended */
  default:
    throw std::invalid_argument("Unknown type for the optimizer");
  }
}

} // namespace nntrainer
