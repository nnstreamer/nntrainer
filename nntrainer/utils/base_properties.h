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
#include <iostream>
#include <sstream>
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

struct int_prop_tag {};       /**< property is treated as integer */
struct vector_prop_tag {};    /**< property is treated as vector, eg) 1,2,3 */
struct dimension_prop_tag {}; /**< property is treated as dimension, eg 1:2:3 */
struct double_prop_tag {};    /**< property is treated as double */
struct str_prop_tag {};       /**< property is treated as string */

/**
 * @brief base property class, inherit this to make a convinient property
 *
 * @tparam T
 */
template <typename T> class Property {

public:
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
  virtual bool is_valid(const T &v) { return true; }

private:
  T value; /**< underlying data */
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
 */
template <typename Tag> struct str_converter {};

/**
 * @copydoc template<typename Tag> struct_converter;
 */
template <> struct str_converter<str_prop_tag> {
  static std::string to_string(const std::string &value) { return value; }

  static std::string from_string(const std::string &str) { return str; }
};

/**
 * @copydoc template<typename Tag> struct_converter;
 */
template <> struct str_converter<int_prop_tag> {
  static std::string to_string(const int value) {
    return std::to_string(value);
  }

  static int from_string(const std::string &value) { return std::stoi(value); }
};

/** convert dispatcher */
template <typename T> std::string to_string(const T &property) {
  using tag_type = typename tag_cast<typename prop_tag<T>::type,
                                     vector_prop_tag, int_prop_tag>::type;
  return str_converter<tag_type>::to_string(property.get());
}

template <typename T> void from_string(const std::string &str, T &property) {
  using tag_type =
    typename tag_cast<typename prop_tag<T>::type, int_prop_tag>::type;
  property.set(str_converter<tag_type>::from_string(str));
}

} // namespace nntrainer

#endif // __BASE_PROPERTIES_H__
