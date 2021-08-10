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

#include <array>
#include <base_properties.h>
#include <fstream>

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
 * @brief trainable property, use this to set and check how if certain layer is
 * trainable
 *
 */
class Trainable : public nntrainer::Property<bool> {
public:
  /**
   * @brief Construct a new Trainable object
   *
   */
  Trainable(bool val = true) : nntrainer::Property<bool>(val) {}
  static constexpr const char *key = "trainable";
  using prop_tag = bool_prop_tag;
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

/**
 * @brief Padding2D property, this is used to calculate padding2D
 * @details Padding2D is saved as a string. Upon calling Padding2D::compute,
 * returns std::vector<unsigned int> which has computed padding2Ds, below
 * formats are accepted valid
 * 1. "same" (case insensitive literal string)
 * 2. "valid" (case insensitive literal string)
 * 3. "padding2D_all", eg) padding=1
 * 4. "padding2D_height, padding2D_width" eg) padding=1,1
 * 5. "padding2D_top, padding2D_bottom, padding2D_left, padding2D_right" eg)
 * padding=1,1,1,1
 *
 */
class Padding2D final : public nntrainer::Property<std::string> {
public:
  /**
   * @brief Construct a new Padding2D object
   *
   */
  Padding2D(const std::string &value = "valid") :
    nntrainer::Property<std::string>(value) {} /**< default value if any */
  bool isValid(const std::string &v) const override;
  using prop_tag = str_prop_tag; /**< property type */

  /**
   * @brief compute actual padding2D from the underlying data
   *
   * @param input input dimension
   * @param kernel kernel dimension
   * @return std::array<unsigned int, 4> list of unsigned padding
   */
  std::array<unsigned int, 4> compute(const TensorDim &input,
                                      const TensorDim &kernel);
};

/**
 * @brief DropOutSpec property, this defines drop out specification of layer
 *
 */
class DropOutSpec : public nntrainer::Property<float> {

public:
  /**
   * @brief Construct a new DropOut object with a default value 0.0
   *
   */
  DropOutSpec(float value = 0.0) : nntrainer::Property<float>(value) {}
  static constexpr const char *key = "dropout"; /**< unique key to access */
  using prop_tag = float_prop_tag;              /**< property type */

  /**
   * @brief DropOutSpec validator
   *
   * @param v float to validate
   * @retval true if it is greater or equal than 0.0
   * @retval false if it is samller than 0.0
   */
  bool isValid(const float &v) const override;
};

/**
 * @brief Props containing file path value
 *
 */
class FilePath : public Property<std::string> {
public:
  /**
   * @brief Construct a new File Path object
   */
  FilePath() : Property<std::string>() {}

  /**
   * @brief Construct a new File Path object
   *
   * @param path path to set
   */
  FilePath(const std::string &path) { set(path); }
  static constexpr const char *key = "path"; /**< unique key to access */
  using prop_tag = str_prop_tag;             /**< property type */

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @return bool true if valid
   */
  bool isValid(const std::string &v) const override;

  /**
   * @brief setter
   *
   * @param v value to set
   */
  void set(const std::string &v) override;

  /**
   * @brief return file size
   *
   * @return std::ifstream::pos_type size of the file
   */
  std::ifstream::pos_type file_size();

private:
  std::ifstream::pos_type cached_pos_size;
};
} // namespace props
} // namespace nntrainer

#endif // __COMMON_PROPERTIES_H__
