// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file        connection.h
 * @date        23 Nov 2020
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       Connection class and related utility functions
 */
#ifndef __CONNECTION_H__
#define __CONNECTION_H__

#include <string>
#include <utility>

namespace nntrainer {
/**
 * @brief RAII class to define a connection
 * @note connection is a light weight class wraps around connection information
 *
 */
class Connection {
public:
  /**
   * @brief Construct a new Connection object
   *
   * @param layer_name layer identifier
   * @param idx index denotes nth tensor in a layer
   */
  Connection(const std::string &layer_name, unsigned int idx);

  /**
   * @brief Construct a new Connection object from string representation
   * string representation is format of {layer_name, idx};
   *
   * @param string_representation string format of {layer_name}({idx})
   */
  explicit Connection(const std::string &string_representation);

  /**
   * @brief Construct a new Connection object
   *
   * @param rhs rhs to copy
   */
  Connection(const Connection &rhs);

  /**
   * @brief Copy assignment operator
   *
   * @param rhs rhs to copy
   * @return Connection&
   */
  Connection &operator=(const Connection &rhs);

  /**
   * @brief Move Construct Connection object
   *
   * @param rhs rhs to move
   */
  Connection(Connection &&rhs) noexcept;

  /**
   * @brief Move assign a connection operator
   *
   * @param rhs rhs to move
   * @return Connection&
   */
  Connection &operator=(Connection &&rhs) noexcept;

  /**
   * @brief string representation of connection
   *
   * @return std::string string format of {name}({idx})
   */
  std::string toString() const;

  /**
   * @brief Get the index
   *
   * @return unsigned index
   */
  const unsigned getIndex() const { return index; }

  /**
   * @brief Get the index
   *
   * @return unsigned index
   */
  unsigned &getIndex() { return index; }

  /**
   * @brief Get the Layer name object
   *
   * @return const Name& name of layer
   */
  const std::string &getName() const { return name; }

  /**
   * @brief Get the Layer name object
   *
   * @return Name& name of layer
   */
  std::string &getName() { return name; }

  /**
   *
   * @brief operator==
   *
   * @param rhs right side to compare
   * @return true if equal
   * @return false if not equal
   */
  bool operator==(const Connection &rhs) const noexcept;

private:
  unsigned index;
  std::string name;
};

} // namespace nntrainer

/**
 * @brief hash specialization for connection
 *
 */
template <> struct std::hash<nntrainer::Connection> {
  /**
   * @brief hash operator
   *
   * @param c connection to hash
   * @return std::size_t hash
   */
  std::size_t operator()(const nntrainer::Connection &c) const {
    return std::hash<std::string>{}(c.toString());
  }
};

#endif // __CONNECTION_H__
