// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   common_properties.cpp
 * @date   14 May 2021
 * @brief  This file contains implementation of common properties widely used
 * across layers
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <common_properties.h>

#include <nntrainer_error.h>
#include <parse_util.h>

#include <regex>
#include <vector>

namespace nntrainer {
namespace props {
bool Name::isValid(const std::string &v) const {

  static std::regex allowed("[a-zA-Z0-9][-_./a-zA-Z0-9]*");
  return std::regex_match(v, allowed);
}

} // namespace props

} // namespace nntrainer
