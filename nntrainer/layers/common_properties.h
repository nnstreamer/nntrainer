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

#include <base_properties.h>

#ifndef __COMMON_PROPERTIES_H__
#define __COMMON_PROPERTIES_H__

namespace nntrainer {
namespace props {

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

  bool is_valid(const unsigned int &v) override { return v > 0; }
};

} // namespace props
} // namespace nntrainer

#endif // __COMMON_PROPERTIES_H__
