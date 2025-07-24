// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   utils.h
 * @date   08 Jan 2021
 * @brief  This file contains simple utilities used across the application
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <string>

namespace simpleshot {

namespace util {

struct Entry {
  std::string key;
  std::string value;
};

Entry getKeyValue(const std::string &input);

} // namespace util
} // namespace simpleshot
