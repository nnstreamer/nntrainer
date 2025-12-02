// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   utils.cpp
 * @date   08 Jan 2021
 * @brief  This file contains simple utilities used across the application
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <iostream>
#include <regex>

#include <simpleshot_utils.h>

namespace simpleshot {

namespace util {

Entry getKeyValue(const std::string &input) {
  Entry entry;
  static const std::regex words_regex("[^\\s=]+");

  std::string input_str(input);
  input_str.erase(std::remove(input_str.begin(), input_str.end(), ' '),
                  input_str.end());
  auto words_begin =
    std::sregex_iterator(input_str.begin(), input_str.end(), words_regex);
  auto words_end = std::sregex_iterator();

  int nwords = std::distance(words_begin, words_end);
  if (nwords != 2) {
    throw std::invalid_argument("key, value is not found");
  }

  entry.key = words_begin->str();
  entry.value = (++words_begin)->str();
  return entry;
}

} // namespace util
} // namespace simpleshot
