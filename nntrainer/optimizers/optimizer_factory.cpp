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
#include <algorithm>
#include <sstream>

#include <adam.h>
#include <nntrainer_error.h>
#include <optimizer_factory.h>
#include <parse_util.h>
#include <sgd.h>

namespace nntrainer {

/// helper function to convert enum to string
/// @todo this should be integrated into appcontext
const std::string optimizerIntToStrType(const OptType &type) {
  switch (type) {
  case OptType::ADAM:
    return "adam";
  case OptType::SGD:
    return "sgd";
  case OptType::UNKNOWN:
  /// fall through intended
  default:
    throw exception::not_supported(
      "[opt_integer_to_string_type] Not supported type given");
  }

  throw exception::not_supported(
    "[opt_integer_to_string_type] Not supported type given");
}
/**
 * @brief Factory creator with copy constructor
 */
std::unique_ptr<Optimizer> createOptimizer(const std::string &type,
                                           const Optimizer &opt) {
  /// #673: use context to create optimizer
  if (istrequal(type, "sgd")) {
    return std::make_unique<SGD>(static_cast<const SGD &>(opt));
  }

  if (istrequal(type, "adam")) {
    return std::make_unique<Adam>(static_cast<const Adam &>(opt));
  }

  std::stringstream ss;
  ss << "Unknown type for the optimizer, type: " << type;

  throw std::invalid_argument(ss.str().c_str());
}

std::unique_ptr<Optimizer> createOptimizer(const OptType &type,
                                           const Optimizer &opt) {
  const std::string &s = optimizerIntToStrType(type);
  return createOptimizer(s, opt);
}

/**
 * @brief Factory creator with constructor
 */
std::unique_ptr<Optimizer> createOptimizer(const std::string &type) {
  /// #673: use context to create optimizer
  if (istrequal(type, "sgd")) {
    return std::make_unique<SGD>();
  }

  if (istrequal(type, "adam")) {
    return std::make_unique<Adam>();
  }

  std::stringstream ss;
  ss << "Unknown type for the optimizer, type: " << type;

  throw std::invalid_argument(ss.str().c_str());
}

std::unique_ptr<Optimizer> createOptimizer(const OptType &type) {
  const std::string &actual_type = optimizerIntToStrType(type);
  return createOptimizer(actual_type);
}

} // namespace nntrainer
