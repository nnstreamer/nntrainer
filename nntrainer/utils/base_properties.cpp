// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file base_properties.cpp
 * @date 03 May 2021
 * @brief Convenient property type definition for automated serialization
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <base_properties.h>
#include <parse_util.h>
#include <tensor_dim.h>

#include <regex>
#include <sstream>
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
std::string str_converter<dimension_prop_tag, TensorDim>::to_string(
  const TensorDim &dimension) {
  std::stringstream ss;
  ss << dimension.batch() << ':' << dimension.channel() << ':'
     << dimension.height() << ':' << dimension.width();
  return ss.str();
}

template <>
TensorDim str_converter<dimension_prop_tag, TensorDim>::from_string(
  const std::string &value) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream iss(value);

  while (std::getline(iss, token, ':')) {
    tokens.push_back(token);
  }

  NNTR_THROW_IF(tokens.size() > MAXDIM, std::invalid_argument)
    << "More than 4 axes is not supported, target string: " << value;

  TensorDim target;

  int cur_axis = 3;
  for (auto iter = tokens.rbegin(); iter != tokens.rend(); iter++) {
    target.setTensorDim(cur_axis--, std::stoul(*iter));
  }

  return target;
}

} // namespace nntrainer
