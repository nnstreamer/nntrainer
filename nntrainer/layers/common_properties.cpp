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
#include <sstream>
#include <utility>
#include <vector>

namespace nntrainer {
namespace props {
bool Name::isValid(const std::string &v) const {
  static std::regex allowed("[a-zA-Z0-9][-_./a-zA-Z0-9]*");
  return !v.empty() && std::regex_match(v, allowed);
}

ConnectionSpec::ConnectionSpec(const std::vector<props::Name> &layer_ids_,
                               const std::string &op_type_) :
  op_type(op_type_),
  layer_ids(layer_ids_) {
  NNTR_THROW_IF((op_type != ConnectionSpec::NoneType && layer_ids.size() < 2),
                std::invalid_argument)
    << "connection type is not none but has only a single or empty layer id, "
       "type: "
    << op_type << " number of names: " << layer_ids.size();

  NNTR_THROW_IF((op_type == ConnectionSpec::NoneType && layer_ids.size() >= 2),
                std::invalid_argument)
    << "connection type is none but has only a single or empty layer id, "
       "number of names: "
    << layer_ids.size();
}

ConnectionSpec::ConnectionSpec(const ConnectionSpec &rhs) = default;
ConnectionSpec &ConnectionSpec::operator=(const ConnectionSpec &rhs) = default;
ConnectionSpec::ConnectionSpec(ConnectionSpec &&rhs) noexcept = default;
ConnectionSpec &ConnectionSpec::
operator=(ConnectionSpec &&rhs) noexcept = default;

bool ConnectionSpec::operator==(const ConnectionSpec &rhs) const {
  return op_type == rhs.op_type && layer_ids == rhs.layer_ids;
}

bool InputSpec::isValid(const ConnectionSpec &v) const {
  return v.getLayerIds().size() > 0;
}

std::string ConnectionSpec::NoneType = "";

} // namespace props

static const std::vector<std::pair<char, std::string>>
  connection_supported_tokens = {{',', "concat"}, {'+', "addition"}};

template <>
std::string
str_converter<props::connection_prop_tag, props::ConnectionSpec>::to_string(
  const props::ConnectionSpec &value) {

  auto &type = value.getOpType();

  if (type == props::ConnectionSpec::NoneType) {
    return value.getLayerIds().front();
  }

  auto &cst = connection_supported_tokens;

  auto find_token = [&type](const std::pair<char, std::string> &token) {
    return token.second == type;
  };

  auto token = std::find_if(cst.begin(), cst.end(), find_token);

  NNTR_THROW_IF(token == cst.end(), std::invalid_argument)
    << "Unsupported type given: " << type;

  std::stringstream ss;
  auto last_iter = value.getLayerIds().end() - 1;
  for (auto iter = value.getLayerIds().begin(); iter != last_iter; ++iter) {
    ss << static_cast<std::string>(*iter) << token->first;
  }
  ss << static_cast<std::string>(*last_iter);

  return ss.str();
}

template <>
props::ConnectionSpec
str_converter<props::connection_prop_tag, props::ConnectionSpec>::from_string(
  const std::string &value) {
  auto generate_regex = [](char token) {
    std::stringstream ss;
    ss << "\\s*\\" << token << "\\s*";

    return std::regex(ss.str());
  };

  auto generate_name_vector = [](const std::vector<std::string> &values) {
    props::Name n;
    std::vector<props::Name> names_;
    names_.reserve(values.size());

    for (auto &item : values) {
      if (!n.isValid(item)) {
        break;
      }
      names_.emplace_back(item);
    }

    return names_;
  };

  for (auto &token : connection_supported_tokens) {
    auto reg_ = generate_regex(token.first);
    auto values = split(value, reg_);
    if (values.size() == 1) {
      continue;
    }

    auto names = generate_name_vector(values);
    if (names.size() == values.size()) {
      return props::ConnectionSpec(names, token.second);
    }
  }

  props::Name n;
  n.set(value); // explicitly trigger validation using set method
  return props::ConnectionSpec({n});
}

} // namespace nntrainer
