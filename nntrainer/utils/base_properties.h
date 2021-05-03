// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file base_properties.h
 * @date 08 April 2021
 * @brief Convenient property type definition for automated serialization
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <nntrainer_error.h>
#include <string>

#ifndef __BASE_PROPERTIES_H__
#define __BASE_PROPERTIES_H__

/** base and predefined structures */

namespace nntrainer {

/**
 * @brief property tag to specialize functions based on this
 *
 * @tparam T property type
 */
template <typename T> struct prop_tag { using type = typename T::prop_tag; };

/**
 * @brief property is treated as integer
 *
 */
struct int_prop_tag {};

/**
 * @brief property is treated as unsigned integer
 *
 */
struct uint_prop_tag {};

/**
 * @brief property is treated as vector, eg) 1,2,3
 *
 */
struct vector_prop_tag {};

/**
 * @brief property is treated as dimension, eg 1:2:3
 *
 */
struct dimension_prop_tag {};

/**
 * @brief property is treated as double
 *
 */
struct double_prop_tag {};

/**
 * @brief property is treated as string
 *
 */
struct str_prop_tag {};

/**
 * @brief base property class, inherit this to make a convenient property
 *
 * @tparam T
 */
template <typename T> class Property {

public:
  /**
   * @brief Construct a new Property object
   *
   */
  Property() = default;

  /**
   * @brief Construct a new Property object, setting default skip validation on
   * purpose
   *
   * @param value default value
   */
  Property(const T &value) : value(value){};

  /**
   * @brief Destroy the Property object
   *
   */
  virtual ~Property() = default;

  /**
   * @brief get the underlying data
   *
   * @return const T& data
   */
  const T &get() const { return value; }

  /**
   * @brief get the underlying data
   *
   * @return T& data
   */
  T &get() { return value; }

  /**
   * @brief set the underlying data
   *
   * @param v value to set
   * @throw std::invalid_argument if argument is not valid
   */
  void set(const T &v) {
    NNTR_THROW_IF(is_valid(v) == false, std::invalid_argument)
      << "argument is not valid";
    value = v;
  }

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @return true if valid
   * @return false if not valid
   */
  virtual bool is_valid(const T &v) { return true; }

private:
  T value; /**< underlying data */
};

/**
 * @brief base string property class, string needs special move / copy semantics
 * because some c++14 compiler does not have nothrow_move_assignment/ctor
 * operator for std::string. This class mitigates the issue
 * As std::string is using default allocator for basic_string, this workaround
 * is safe
 *
 */
template <> class Property<std::string> {

public:
  /**
   * @brief Construct a new Property object
   *
   */
  Property() = default;

  /**
   * @brief Construct a new Property object, setting default skip validation on
   * purpose
   *
   * @param value default value
   */
  Property(const std::string &value) : value(value){};

  Property(const Property &rhs) = default;
  Property &operator=(const Property &) = default;

  Property(Property &&rhs) noexcept = default;

  /**
   * @brief move assignment operator, this patch makes,
   * std::is_nothrow_move_assignable<Property<std::string>>::value == true
   * Which might not hold true for some of the old compilers.
   *
   * @param rhs rvalue property to move
   * @return Property& moved result
   */
  Property &operator=(Property &&rhs) noexcept {
    value = std::move(rhs.value);
    return *this;
  }

  /**
   * @brief Destroy the Property object
   *
   */
  virtual ~Property() = default;

  /**
   * @brief get the underlying data
   *
   * @return const T& data
   */
  const std::string &get() const { return value; }

  /**
   * @brief get the underlying data
   *
   * @return T& data
   */
  std::string &get() { return value; }

  /**
   * @brief set the underlying data
   *
   * @param v value to set
   * @throw std::invalid_argument if argument is not valid
   */
  void set(const std::string &v) {
    if (!is_valid(v)) {
      throw std::invalid_argument("argument is not valid");
    }
    value = v;
  }

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @return true if valid
   * @return false if not valid
   */
  virtual bool is_valid(const std::string &v) { return true; }

private:
  std::string value; /**< underlying data */
};

/**
 * @brief meta function to cast tag to it's base
 * @code below is the test spec for the cast
 *
 * struct custom_tag: int_prop_tag {};
 *
 * using tag_type = tag_cast<custom_tag, vector_prop_tag>::type
 * static_assert(<std::is_save_v<tag_type, custom_tag> == false);
 *
 * using tag_type = tag_cast<custom_tag, int_prop_tag>::type
 * static_assert(<std::is_save_v<tag_type, int_prop_tag> == true);
 *
 * using tag_type = tag_cast<custom_tag, vector_prop_tag, int_prop_tag>::type
 * static_assert(std::is_same_v<tag_type, int_prop_tag> == true);
 *
 * @tparam Tags First tag: tag to be casted, rest tags: candidates
 */
template <typename... Tags> struct tag_cast;

/**
 * @brief base case of tag_cast, if nothing matches return @a Tag
 *
 * @tparam Tag Tag to be casted
 * @tparam Others empty parameter pack
 */
template <typename Tag, typename... Others> struct tag_cast<Tag, Others...> {
  using type = Tag;
};

/**
 * @brief normal case of the tag cast
 *
 * @tparam Tag tag to be casted
 * @tparam BaseTag candidates to cast the tag
 * @tparam Others pending candidates to be compared
 */
template <typename Tag, typename BaseTag, typename... Others>
struct tag_cast<Tag, BaseTag, Others...> {
  using type = std::conditional_t<std::is_base_of<BaseTag, Tag>::value, BaseTag,
                                  typename tag_cast<Tag, Others...>::type>;
};

/**
 * @brief property to string converter.
 * This structure defines how to convert to convert from/to string
 *
 * @tparam Tag tag type for the converter
 * @tparam DataType underlying datatype
 */
template <typename Tag, typename DataType> struct str_converter {

  /**
   * @brief convert underlying value to string
   *
   * @param value value to convert to string
   * @return std::string string
   */
  static std::string to_string(const DataType &value);

  /**
   * @brief convert string to underlying value
   *
   * @param value value to convert to string
   * @return DataType converted type
   */
  static DataType from_string(const std::string &value);
};

/**
 * @brief convert dispatcher (to string)
 *
 * @tparam T type to convert
 * @param property property to convert
 * @return std::string converted string
 */
template <typename T> std::string to_string(const T &property) {
  using tag_type =
    typename tag_cast<typename prop_tag<T>::type, int_prop_tag, uint_prop_tag,
                      vector_prop_tag, dimension_prop_tag, double_prop_tag,
                      str_prop_tag>::type;

  using data_type = std::remove_cv_t<
    std::remove_reference_t<decltype(std::declval<T>().get())>>;
  return str_converter<tag_type, data_type>::to_string(property.get());
}

/**
 * @brief convert dispatcher (from string)
 *
 * @tparam T type to convert
 * @param str string to convert
 * @param[out] property property, converted type
 */
template <typename T> void from_string(const std::string &str, T &property) {
  using tag_type =
    typename tag_cast<typename prop_tag<T>::type, int_prop_tag, uint_prop_tag,
                      vector_prop_tag, dimension_prop_tag, double_prop_tag,
                      str_prop_tag>::type;

  using data_type = std::remove_cv_t<
    std::remove_reference_t<decltype(std::declval<T>().get())>>;

  property.set(str_converter<tag_type, data_type>::from_string(str));
}

} // namespace nntrainer

#endif // __BASE_PROPERTIES_H__
