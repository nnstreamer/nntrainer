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

#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace nntrainer {
bool PositiveIntegerProperty::isValid(const unsigned int &value) const {
  return value > 0;
}

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
std::string str_converter<bool_prop_tag, bool>::to_string(const bool &value) {
  return value ? "true" : "false";
}

template <>
bool str_converter<bool_prop_tag, bool>::from_string(const std::string &value) {
  if (value == "true") {
    return true;
  }
  if (value == "false") {
    return false;
  }
  std::string error_msg = "converting value to boolean failed, value: " + value;
  throw std::invalid_argument(error_msg);
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
str_converter<size_t_prop_tag, size_t>::to_string(const size_t &value) {
  return std::to_string(value);
}

template <>
size_t
str_converter<size_t_prop_tag, size_t>::from_string(const std::string &value) {
  return std::stoul(value);
}

template <>
std::string
str_converter<float_prop_tag, float>::to_string(const float &value) {
  return std::to_string(value);
}

template <>
float str_converter<float_prop_tag, float>::from_string(
  const std::string &value) {
  return std::stof(value);
}

template <>
std::string
str_converter<double_prop_tag, double>::to_string(const double &value) {
  std::ostringstream ss;
  ss << value;
  return ss.str();
}

template <>
double
str_converter<double_prop_tag, double>::from_string(const std::string &value) {
  return std::stod(value);
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

  NNTR_THROW_IF(tokens.size() > ml::train::TensorDim::MAXDIM,
                std::invalid_argument)
    << "More than 4 axes is not supported, target string: " << value;

  TensorDim target;

  int cur_axis = 3;
  for (auto iter = tokens.rbegin(); iter != tokens.rend(); iter++) {
    target.setTensorDim(cur_axis--, std::stoul(*iter));
  }

  return target;
}

} // namespace nntrainer
