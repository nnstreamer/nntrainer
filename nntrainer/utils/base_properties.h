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
#include <array>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <nntrainer_error.h>
#include <parse_util.h>

#ifndef __BASE_PROPERTIES_H__
#define __BASE_PROPERTIES_H__

/** base and predefined structures */

namespace nntrainer {

template <typename T>
using remove_cv_ref_t = std::remove_cv_t<std::remove_reference_t<T>>;
/**
 * @brief property info to specialize functions based on this
 * @tparam T property type
 */
template <typename T> struct prop_info {
  using prop_type = remove_cv_ref_t<T>;          /** property type of T */
  using tag_type = typename prop_type::prop_tag; /** Property tag of T */
  using data_type = remove_cv_ref_t<decltype(
    std::declval<prop_type>().get())>; /** Underlying datatype of T */
};

/**
 * @brief property info when it is wrapped inside a vector
 *
 * @tparam T property type
 */
template <typename T> struct prop_info<std::vector<T>> {
  using prop_type = typename prop_info<T>::prop_type;
  using tag_type = typename prop_info<T>::tag_type;
  using data_type = typename prop_info<T>::data_type;
};

/**
 * @brief property info when it is wrapped inside an array
 *
 * @tparam T property type
 */
template <typename T, size_t size> struct prop_info<std::array<T, size>> {
  using prop_type = typename prop_info<T>::prop_type;
  using tag_type = typename prop_info<T>::tag_type;
  using data_type = typename prop_info<T>::data_type;
};

/**
 * @brief Get the Prop Key object
 *
 * @tparam T property to get type
 * @param prop property
 * @return constexpr const char* key
 */
template <typename T> constexpr const char *getPropKey(T &&prop) {
  return prop_info<remove_cv_ref_t<T>>::prop_type::key;
}

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
 * @brief property is treated as dimension, eg 1:2:3
 *
 */
struct dimension_prop_tag {};

/**
 * @brief property is treated as float
 *
 */
struct float_prop_tag {};

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
  Property() : value(nullptr){};

  /**
   * @brief Construct a new Property object
   *
   * @param value default value
   */
  Property(const T &value_) { set(value_); }

  /**
   * @brief Copy Construct a new Property object
   *
   * @param rhs right side to copy from
   */
  Property(const Property &rhs) {
    if (this != &rhs) {
      value = std::make_unique<T>(*rhs.value);
    }
  }

  /**
   * @brief Copy assignment operator of a new property
   *
   * @param rhs right side to copy from
   * @return Property& this
   */
  Property &operator=(const Property &rhs) {
    if (this != &rhs) {
      value = std::make_unique<T>(*rhs.value);
    }
    return *this;
  };

  Property(Property &&rhs) noexcept = default;
  Property &operator=(Property &&rhs) noexcept = default;

  /**
   * @brief Destroy the Property object
   *
   */
  virtual ~Property() = default;

  /**
   * @brief cast operator for property
   *
   * @return T value
   */
  operator T &() { return get(); }

  /**
   * @brief cast operator for property
   *
   * @return T value
   */
  operator const T &() const { return get(); }

  /**
   * @brief get the underlying data
   *
   * @return const T& data
   */
  const T &get() const {
    NNTR_THROW_IF(value == nullptr, std::invalid_argument)
      << "Cannot get property, property is empty";
    return *value;
  }

  /**
   * @brief get the underlying data
   *
   * @return T& data
   */
  T &get() {
    NNTR_THROW_IF(value == nullptr, std::invalid_argument)
      << "Cannot get property, property is empty";
    return *value;
  }

  /**
   * @brief check if property is empty
   *
   * @retval true empty
   * @retval false not empty
   */
  bool empty() const { return value == nullptr; }

  /**
   * @brief set the underlying data
   *
   * @param v value to set
   * @throw std::invalid_argument if argument is not valid
   */
  void set(const T &v) {
    NNTR_THROW_IF(isValid(v) == false, std::invalid_argument)
      << "argument is not valid";
    value = std::make_unique<T>(v);
  }

  /**
   * @brief check if given value is valid
   *
   * @param v value to check
   * @retval true if valid
   * @retval false if not valid
   */
  virtual bool isValid(const T &v) const { return true; }

  /**
   * @brief operator==
   *
   * @param rhs right side to compare
   * @retval true if equal
   * @retval false if not equal
   */
  bool operator==(const Property<T> &rhs) const { return *value == *rhs.value; }

private:
  std::unique_ptr<T> value; /**< underlying data */
};

/**
 * @brief meta function to cast tag to it's base
 * @code below is the test spec for the cast
 *
 * struct custom_tag: int_prop_tag {};
 *
 * using tag_type = tag_cast<custom_tag, float_prop_tag>::type
 * static_assert(<std::is_same_v<tag_type, custom_tag> == true);
 *
 * using tag_type = tag_cast<custom_tag, int_prop_tag>::type
 * static_assert(<std::is_same_v<tag_type, int_prop_tag> == true);
 *
 * using tag_type = tag_cast<custom_tag, float_prop_tag, int_prop_tag>::type
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
  using info = prop_info<T>;
  using tag_type =
    typename tag_cast<typename info::tag_type, int_prop_tag, uint_prop_tag,
                      dimension_prop_tag, float_prop_tag, str_prop_tag>::type;
  using data_type = typename info::data_type;

  return str_converter<tag_type, data_type>::to_string(property.get());
}

/**
 * @brief to_string vector specialization
 * @copydoc template <typename T> std::string to_string(const T &property)
 */
template <typename T> std::string to_string(const std::vector<T> &property) {
  std::stringstream ss;
  auto last_iter = property.end() - 1;
  for (auto iter = property.begin(); iter != last_iter; ++iter) {
    ss << to_string(*iter) << ',';
  }
  ss << to_string(*last_iter);

  return ss.str();
}

/**
 * @brief to_string array specialization
 * @copydoc template <typename T> std::string to_string(const T &property)
 */
template <typename T, size_t sz>
static std::string to_string(const std::array<T, sz> &value) {
  std::stringstream ss;
  auto last_iter = value.end() - 1;
  for (auto iter = value.begin(); iter != last_iter; ++iter) {
    ss << to_string(*iter) << ',';
  }
  ss << to_string(*last_iter);

  return ss.str();
}

/**
 *
 * @brief convert dispatcher (from string)
 *
 *
 * @tparam T type to convert
 * @param str string to convert
 * @param[out] property property, converted type
 */
template <typename T> void from_string(const std::string &str, T &property) {
  using info = prop_info<T>;
  using tag_type =
    typename tag_cast<typename info::tag_type, int_prop_tag, uint_prop_tag,
                      dimension_prop_tag, float_prop_tag, str_prop_tag>::type;
  using data_type = typename info::data_type;

  property.set(str_converter<tag_type, data_type>::from_string(str));
}

/**
 * @brief transform iternal data, this is to use with std::transform
 *
 * @param item item to transform
 * @return DataType transformed result
 */
template <typename T> static T from_string_helper_(const std::string &item) {
  T t;
  from_string(item, t);
  return t;
}

static const std::regex reg_("\\s*\\,\\s*");

/**
 * @brief from_string array specialization
 * @copydoc template <typename T> void from_string(const std::string &str, T
 * &property)
 * @note array implies that the size is @b fixed so there will be a validation
 * check on size
 */
template <typename T, size_t sz>
void from_string(const std::string &value, std::array<T, sz> &property) {
  auto v = split(value, reg_);
  NNTR_THROW_IF(v.size() != sz, std::invalid_argument)
    << "size must match with array size, array size: " << sz
    << " string: " << value;

  std::transform(v.begin(), v.end(), property.begin(), from_string_helper_<T>);
}

/**
 * @brief from_string vector specialization
 * @copydoc str_converter<Tag, DataType>::to_string(const DataType &value)
 * @note vector implies that the size is @b not fixed so there shouldn't be any
 * validation on size
 *
 */
template <typename T>
void from_string(const std::string &value, std::vector<T> &property) {
  auto v = split(value, reg_);

  property.clear();
  property.reserve(v.size());
  std::transform(v.begin(), v.end(), std::back_inserter(property),
                 from_string_helper_<T>);
}

} // namespace nntrainer

#endif // __BASE_PROPERTIES_H__
