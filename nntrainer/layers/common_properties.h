// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file common_properties.h
 * @date 09 April 2021
 * @brief This file contains list of common properties widely used across layers
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <string>

#include <base_properties.h>

#ifndef __COMMON_PROPERTIES_H__
#define __COMMON_PROPERTIES_H__

namespace nntrainer {
namespace props {

/**
 * @brief Name property, name is an identifier of an object
 *
 */
class Name : public nntrainer::Property<std::string> {
public:
  Name(const std::string &value = "") :
    nntrainer::Property<std::string>(value) {} /**< default value if any */
  static constexpr const char *key = "name";   /**< unique key to access */
  using prop_tag = str_prop_tag;               /**< property type */
};

/**
 * @brief unit property, unit is used to measure how many weights are there
 *
 */
class Unit : public nntrainer::Property<unsigned int> {
public:
  Unit(unsigned int value = 0) :
    nntrainer::Property<unsigned int>(value) {} /**< default value if any */
  static constexpr const char *key = "unit";    /**< unique key to access */
  using prop_tag = uint_prop_tag;               /**< property type */

  bool isValid(const unsigned int &v) const override { return v > 0; }
};

} // namespace props
} // namespace nntrainer

#endif // __COMMON_PROPERTIES_H__
