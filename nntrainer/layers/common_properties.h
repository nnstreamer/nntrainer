// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   common_properties.h
 * @date   09 April 2021
 * @brief  This file contains list of common properties widely used across
 * layers
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <string>

#include <base_properties.h>

#ifndef __COMMON_PROPERTIES_H__
#define __COMMON_PROPERTIES_H__

namespace nntrainer {
namespace props {

/**
 * @brief Name property, name is an identifier of an object
 *
 */
class Name : public nntrainer::Property<std::string> {
public:
  /**
   * @brief Construct a new Name object without a default value
   *
   */
  Name() : nntrainer::Property<std::string>() {}

  /**
   * @brief Construct a new Name object with a default value
   *
   * @param value value to contrusct the property
   */
  Name(const std::string &value) : nntrainer::Property<std::string>(value) {}
  static constexpr const char *key = "name"; /**< unique key to access */
  using prop_tag = str_prop_tag;             /**< property type */

  /**
   * @brief name validator
   *
   * @param v string to validate
   * @retval true if it contains alphanumeric and/or '-', '_', '/'
   * @retval false if it is empty or contains non-valid character
   */
  bool isValid(const std::string &v) const override;
};

/**
 * @brief unit property, unit is used to measure how many weights are there
 *
 */
class Unit : public nntrainer::Property<unsigned int> {
public:
  Unit(unsigned int value = 1) :
    nntrainer::Property<unsigned int>(value) {} /**< default value if any */
  static constexpr const char *key = "unit";    /**< unique key to access */
  using prop_tag = uint_prop_tag;               /**< property type */

  bool isValid(const unsigned int &v) const override { return v > 0; }
};

/**
 * @brief RAII class to define the connection spec
 *
 */
class ConnectionSpec {
public:
  static std::string NoneType;

  /**
   * @brief Construct a new Connection Spec object
   *
   * @param layer_ids_ layer ids that will be an operand
   * @param op_type_ operator type
   */
  ConnectionSpec(const std::vector<Name> &layer_ids_,
                 const std::string &op_type_ = ConnectionSpec::NoneType);

  /**
   * @brief Construct a new Connection Spec object
   *
   * @param rhs rhs to copy
   */
  ConnectionSpec(const ConnectionSpec &rhs);

  /**
   * @brief Copy assignment operator
   *
   * @param rhs rhs to copy
   * @return ConnectionSpec&
   */
  ConnectionSpec &operator=(const ConnectionSpec &rhs);

  /**
   * @brief Move Construct Connection Spec object
   *
   * @param rhs rhs to move
   */
  ConnectionSpec(ConnectionSpec &&rhs) noexcept;

  /**
   * @brief Move assign a connection spec operator
   *
   * @param rhs rhs to move
   * @return ConnectionSpec&
   */
  ConnectionSpec &operator=(ConnectionSpec &&rhs) noexcept;

  /**
   * @brief Get the Op Type object
   *
   * @return const std::string& op_type (read-only)
   */
  const std::string &getOpType() const { return op_type; }

  /**
   * @brief Get the Layer Ids object
   *
   * @return const std::vector<Name>& vector of layer ids (read-only)
   */
  const std::vector<Name> &getLayerIds() const { return layer_ids; }

  /**
   *
   * @brief operator==
   *
   * @param rhs right side to compare
   * @return true if equal
   * @return false if not equal
   */
  bool operator==(const ConnectionSpec &rhs) const;

private:
  std::string op_type;
  std::vector<Name> layer_ids;
};

/**
 * @brief Connection prop tag type
 *
 */
struct connection_prop_tag {};

/**
 * @brief InputSpec property, this defines connection specification of an input
 *
 */
class InputSpec : public nntrainer::Property<ConnectionSpec> {
public:
  /**
   * @brief Construct a new Input Spec object
   *
   */
  InputSpec() : nntrainer::Property<ConnectionSpec>() {}

  /**
   * @brief Construct a new Input Spec object
   *
   * @param value default value of a input spec
   */
  InputSpec(const ConnectionSpec &value) :
    nntrainer::Property<ConnectionSpec>(value) {} /**< default value if any */
  static constexpr const char *key =
    "input_layers";                     /**< unique key to access */
  using prop_tag = connection_prop_tag; /**< property type */
  bool isValid(const ConnectionSpec &v) const override;
};

} // namespace props
} // namespace nntrainer

#endif // __COMMON_PROPERTIES_H__
