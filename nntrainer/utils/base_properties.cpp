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

#include <cerrno>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace {

/**
 * @brief constructs system error_code from POSIX errno
 */
[[maybe_unused]] auto make_system_error_code() {
  return std::error_code{errno, std::system_category()};
}

bool isFileReadAccessisble(fs::path p, std::error_code &ec) {
#if defined(_POSIX_VERSION)
  // Check if we have read permissions to path pointing this file
  auto r = ::access(p.c_str(), R_OK);
  if (r == 0) {
    ec = std::error_code{};
    return true;
  }

  ec = make_system_error_code();

  return false;
#else
  ec = std::error_code{};
  // Unless it is POSIX, best bet is to try it
  std::ifstream file(p, std::ios::binary | std::ios::ate);
  return file.good();
#endif
}

/**
 * @brief Helper for testing path kind looking behind symlinks.
 */
template <typename FileCheckFn_>
bool isPathKindHelper(const fs::path v, FileCheckFn_ &&file_check_fn) noexcept {
  // Reject empty and non-existing paths
  {
    std::error_code ec;
    if (v.empty() || !exists(v, ec))
      return false;

    if (ec)
      return false;
  }

  // Check if it is a path is of file_check_fn kind
  {
    std::error_code ec;
    auto real_path = is_symlink(v) ? read_symlink(v, ec) : v;

    if (ec)
      return false;

    if (!file_check_fn(real_path, ec))
      return false;

    if (ec)
      return false;
  }

  return true;
}
} // namespace

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
std::string
str_converter<path_prop_tag, fs::path>::to_string(const fs::path &value) {
  return value;
}

template <>
fs::path
str_converter<path_prop_tag, fs::path>::from_string(const std::string &value) {
  return fs::path{value};
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

PathProperty::~PathProperty() = default;

bool PathProperty::isRegularFile(const fs::path &v) noexcept {
  return isPathKindHelper(
    v, [](const auto &v, auto ec) { return fs::is_regular_file(v, ec); });
}

bool PathProperty::isDirectory(const fs::path &v) noexcept {
  return isPathKindHelper(
    v, [](const auto &v, auto ec) { return fs::is_directory(v, ec); });
}

bool PathProperty::isReadAccessible(const fs::path &v) noexcept {
  std::error_code ec;

  if (!isFileReadAccessisble(v, ec))
    return false;

  return !ec;
}

} // namespace nntrainer
