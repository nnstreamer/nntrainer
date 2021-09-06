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
#include <tensor_dim.h>

#include <regex>
#include <sstream>
#include <utility>
#include <vector>

namespace nntrainer {
namespace props {

Name::Name() : nntrainer::Property<std::string>() {}

Name::Name(const std::string &value) { set(value); }

void Name::set(const std::string &value) {
  auto to_lower = [](const std::string &str) {
    std::string ret = str;
    std::transform(ret.begin(), ret.end(), ret.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ret;
  };
  nntrainer::Property<std::string>::set(to_lower(value));
}

bool Name::isValid(const std::string &v) const {
  static std::regex allowed("[a-zA-Z0-9][-_./a-zA-Z0-9]*");
  return !v.empty() && std::regex_match(v, allowed);
}

Normalization::Normalization(bool value) { set(value); }

Standardization::Standardization(bool value) { set(value); }

bool DropOutSpec::isValid(const float &v) const {
  if (v <= 0.0)
    return false;
  else
    return true;
}

bool FilePath::isValid(const std::string &v) const {
  std::ifstream file(v, std::ios::binary | std::ios::ate);
  return file.good();
}

void FilePath::set(const std::string &v) {
  Property<std::string>::set(v);
  std::ifstream file(v, std::ios::binary | std::ios::ate);
  cached_pos_size = file.tellg();
}

std::ifstream::pos_type FilePath::file_size() { return cached_pos_size; }

bool NumClass::isValid(const unsigned int &v) const { return v > 0; }

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

/**
 * @brief unsigned integer property, internally used to parse padding values
 *
 */
class Padding_ : public nntrainer::Property<int> {
public:
  using prop_tag = int_prop_tag; /**< property type */
};

bool Padding2D::isValid(const std::string &v) const {

  /// case 1, 2: padding has string literal
  if (istrequal(v, "valid") || istrequal(v, "same")) {
    return true;
  }

  std::vector<props::Padding_> paddings;
  from_string(v, paddings);

  /// case 3, 4, 5: padding has a sequence of unsigned integer
  if (paddings.size() == 1 || paddings.size() == 2 || paddings.size() == 4) {
    /// check if every padding is non-negative integer
    for (const auto &padding : paddings) {
      if (padding.get() < 0) {
        return false;
      }
    }
    return true;
  }

  /// case else: false
  return false;
}

std::array<unsigned int, 4> Padding2D::compute(const TensorDim &input,
                                               const TensorDim &kernel) {
  auto &padding_repr = get(); /// padding representation
  if (istrequal(padding_repr, "valid")) {
    return {0, 0, 0, 0};
  }

  if (istrequal(padding_repr, "same")) {
    /// @note if we start to consider dilation, this calculation has to tuned
    /// accordingly.
    auto calculate_padding = [](unsigned input_, unsigned kernel_) {
      NNTR_THROW_IF(input_ < kernel_, std::invalid_argument)
        << "input smaller then kernel not supported, input size: " << input_
        << " kernel size: " << kernel_ << " padding: same\n";
      return kernel_ - 1;
    };

    auto pad_horizontal = calculate_padding(input.width(), kernel.width());
    auto pad_vertical = calculate_padding(input.height(), kernel.height());

    auto pad_top = pad_vertical / 2;
    auto pad_left = pad_horizontal / 2;

    return {pad_top, pad_vertical - pad_top, pad_left,
            pad_horizontal - pad_left};
  }

  /// case 3, 4, 5: padding has a sequence of unsigned integer
  std::vector<props::Padding_> paddings_;
  from_string(padding_repr, paddings_);
  std::vector<unsigned int> paddings(paddings_.begin(), paddings_.end());

  switch (paddings.size()) {
  case 1:
    return {paddings[0], paddings[0], paddings[0], paddings[0]};
  case 2:
    return {paddings[0], paddings[0], paddings[1], paddings[1]};
  case 4:
    return {paddings[0], paddings[1], paddings[2], paddings[3]};
  default:
    throw std::logic_error("[padding] should not reach here");
  }

  throw std::logic_error("[padding] should not reach here");
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
