// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file base_properties.h
 * @date 03 May 2021
 * @brief Convenient property type definition for automated serialization
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <base_properties.h>
#include <parse_util.h>

#include <regex>
#include <string>
#include <vector>

namespace nntrainer {

template <>
std::string
str_converter<str_prop_tag, std::string>::to_string(const std::string &value) {
  return value;
}

template <>
std::string str_converter<str_prop_tag, std::string>::from_string(
  const std::string &value) {
  return value;
}

template <>
std::string str_converter<int_prop_tag, int>::to_string(const int &value) {
  return std::to_string(value);
}

template <>
int str_converter<int_prop_tag, int>::from_string(const std::string &value) {
  return std::stoi(value);
}

template <>
std::string str_converter<uint_prop_tag, unsigned int>::to_string(
  const unsigned int &value) {
  return std::to_string(value);
}

template <>
unsigned int str_converter<uint_prop_tag, unsigned int>::from_string(
  const std::string &value) {
  return std::stoul(value);
}

template <>
std::string
str_converter<vector_prop_tag, std::vector<unsigned int>>::to_string(
  const std::vector<unsigned int> &value) {
  std::stringstream ss;
  auto last_iter = value.end() - 1;
  for (auto iter = value.begin(); iter != last_iter; ++iter) {
    ss << *iter << ',';
  }
  ss << *(value.end() - 1);

  return ss.str();
}

template <>
std::vector<unsigned int>
str_converter<vector_prop_tag, std::vector<unsigned int>>::from_string(
  const std::string &value) {
  static const std::regex reg("\\s*\\,\\s*");
  auto v = split(value, reg);

  std::vector<unsigned int> rets;
  rets.reserve(v.size());

  std::transform(v.begin(), v.end(), std::back_inserter(rets),
                 [](const std::string &item) { return std::stoul(item); });

  return rets;
}

template <>
std::string str_converter<vector_prop_tag, std::vector<std::string>>::to_string(
  const std::vector<std::string> &value) {
  std::stringstream ss;
  auto last_iter = value.end() - 1;
  for (auto iter = value.begin(); iter != last_iter; ++iter) {
    ss << *iter << ',';
  }
  ss << *(value.end() - 1);

  return ss.str();
}

template <>
std::vector<std::string>
str_converter<vector_prop_tag, std::vector<std::string>>::from_string(
  const std::string &value) {
  static const std::regex reg("\\s*\\,\\s*");

  return split(value, reg);
  ;
}

} // namespace nntrainer
