// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file        connection.cpp
 * @date        23 Nov 2021
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       Connection class and related utility functions
 */
#include <connection.h>

#include <common_properties.h>
#include <stdexcept>

namespace {}

namespace nntrainer {
Connection::Connection(const std::string &layer_name, unsigned int idx) :
  index(idx),
  name(props::Name(layer_name).get()) {}

Connection::Connection(const std::string &string_representation) {
  auto &sr = string_representation;
  auto pos = sr.find_first_of('(');
  auto idx = 0u;
  auto name_part = sr.substr(0, pos);

  if (pos != std::string::npos) {
    NNTR_THROW_IF(sr.back() != ')', std::invalid_argument)
      << "failed to parse connection invalid format: " << sr;

    auto idx_part = std::string(sr.begin() + pos + 1, sr.end() - 1);
    /// idx_part must not have '(' or ')' inside
    NNTR_THROW_IF(idx_part.find_first_of('(') != std::string::npos or
                    idx_part.find_first_of(')') != std::string::npos,
                  std::invalid_argument)
      << "failed to parse connection invalid format: " << sr;
    idx = str_converter<uint_prop_tag, unsigned>::from_string(idx_part);
  }

  index = idx;
  name = props::Name(name_part);
}

Connection::Connection(const Connection &rhs) = default;
Connection &Connection::operator=(const Connection &rhs) = default;
Connection::Connection(Connection &&rhs) noexcept = default;
Connection &Connection::operator=(Connection &&rhs) noexcept = default;

bool Connection::operator==(const Connection &rhs) const noexcept {
  return index == rhs.index and name == rhs.name;
}

std::string Connection::toString() const {
  std::stringstream ss;
  ss << getName() << '(' << getIndex() << ')';
  return ss.str();
}

}; // namespace nntrainer
