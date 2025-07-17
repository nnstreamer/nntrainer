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

#if defined(_WIN32)
#define NNTR_API __declspec(dllexport)
#else
#define NNTR_API
#endif

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
  NNTR_API Connection(const std::string &layer_name, unsigned int idx);

  /**
   * @brief Construct a new Connection object from string representation
   * string representation is format of {layer_name, idx};
   *
   * @param str_repr string format of {layer_name}({idx})
   */
  NNTR_API explicit Connection(const std::string &str_repr);

  /**
   * @brief Construct a new Connection object
   *
   * @param rhs rhs to copy
   */
  NNTR_API Connection(const Connection &rhs);

  /**
   * @brief Copy assignment operator
   *
   * @param rhs rhs to copy
   * @return Connection&
   */
  NNTR_API Connection &operator=(const Connection &rhs);

  /**
   * @brief Move Construct Connection object
   *
   * @param rhs rhs to move
   */
  NNTR_API Connection(Connection &&rhs) noexcept;

  /**
   * @brief Move assign a connection operator
   *
   * @param rhs rhs to move
   * @return Connection&
   */
  NNTR_API Connection &operator=(Connection &&rhs) noexcept;

  /**
   * @brief string representation of connection
   *
   * @return std::string string format of {name}({idx})
   */
  NNTR_API std::string toString() const;

  /**
   * @brief Get the index
   *
   * @return unsigned index
   */
  NNTR_API const unsigned getIndex() const { return index; }

  /**
   * @brief Get the index
   *
   * @return unsigned index
   */
  NNTR_API unsigned &getIndex() { return index; }

  /**
   * @brief Get the Layer name object
   *
   * @return const Name& name of layer
   */
  NNTR_API const std::string &getName() const { return name; }

  /**
   * @brief Get the Layer name object
   *
   * @return Name& name of layer
   */
  NNTR_API std::string &getName() { return name; }

  /**
   *
   * @brief operator==
   *
   * @param rhs right side to compare
   * @return true if equal
   * @return false if not equal
   */
  NNTR_API bool operator==(const Connection &rhs) const noexcept;

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
  NNTR_API std::size_t operator()(const nntrainer::Connection &c) const {
    return std::hash<std::string>{}(c.toString());
  }
};

#endif // __CONNECTION_H__
