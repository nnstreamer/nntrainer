// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file properties.h
 * @date 03 Aug 2020
 * @brief Properties handler for NNtrainer.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNTRAINER_PROPERTIES_H__
#define __NNTRAINER_PROPERTIES_H__
#ifdef __cplusplus

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <nntrainer_error.h>

namespace nntrainer {
template <typename T> struct prop_traits {
  static constexpr bool is_property = false; /**< check if this is property */

  typedef bool value_type; /**< actual value type of given property */

  /**
   * @brief validation logic for type T
   * @param[in] value Target to have registration
   */
  static bool is_valid(value_type value) {  throw std::runtime_error("no trait type can be validated"); }

  /**
   * @brief get string key of current property
   */
  static std::string getKey() {  throw std::runtime_error("no trait type have a key"); }

  /**
   * @brief get default value of current type
   */
  static value_type getDefault() {
    throw std::runtime_error("no trait type does not have default");
  }

  /**
   * @brief serialization for type T
   * @param[in] target Target to be converted to string
   */
  static std::string serialize(value_type target) {
    throw std::runtime_error("Only property type can be serialized");
  }

  /**
   * @brief deserialization for type T
   * @param[in] str string to be converted to actual type T
   * @throw std::runtime_error if T is not property
   * @throw std::invalid_argument if @a str is not deserializable.
   */
  static value_type deserialize(std::string &str) {
    throw std::runtime_error("Only property type can be deserialized");
  }
};

/*****************************************************************************
 *  Property registration part.                                              *
 *  If you want to add a property, only this part need change.               *
 *  You only have to add property struct and prop_traits.                    *
 *                                                                           *
 *  It is recommened to use unique key for each types.                       *
 *  However, it is allowed to use duplicated key                             *
 *  When those properties are mutually exclusive                             *
 *****************************************************************************/

/**< color is here for demonstration purpose. */

enum class Color {
  black,
  white,
  unknown,
};

/**
 * @brief trait specialization part
 */
template <> struct prop_traits<Color> {

  static constexpr bool is_property = true;

  typedef Color value_type;

  static std::string getKey() { return "color"; }

  static bool is_valid(Color c) { return c != Color::unknown; }

  static Color getDefault() { return Color::unknown; }

  static std::string serialize(Color c) {
    switch (c) {
    case Color::black:
      return "black";
    case Color::white:
      return "white";
    default:
      return "unknown";
    }
  }

  static Color deserialize(const std::string &val) {
    if (val == "black") {
      return Color::black;
    }
    if (val == "white") {
      return Color::white;
    }
    return Color::unknown;
  }
};

/*****************************************************************************
 *  End of property registration part.                                       *
 *****************************************************************************/

/**
 * @brief base class of property holder. This property holder enables set/get
 * access to props by string
 */
class PropertyHolder {

public:
  /**
   * @brief virtual destructor of class
   */
  virtual ~PropertyHolder() {}

  /**
   * @brief setProperty from string representation of value
   * @param[in] value to be deserialized by prop trait.
   * @throw std::invalid_argument if validation fails
   */
  virtual void setProp(const std::string &val) = 0;

  /**
   * @brief getProperty to string representation of value
   * @retval string reprenstation of current value
   */
  virtual std::string getProp() = 0;
};

/**
 * @brief actual holder of property. This class holds pointer of actual type.
 * Which will be residing on Properties::items;
 */
template <typename T> class _PropertyHolder : public PropertyHolder {
  static_assert(prop_traits<T>::is_property, "Only property type can be used");

public:
  typedef typename prop_traits<T>::value_type value_type;
  typedef T prop_type;

  /**
   * @brief property holder destructor
   */
  ~_PropertyHolder() {}

  _PropertyHolder() {}

  value_type get() { return data; }

  void set(value_type _data) {
    if (prop_traits<T>::is_valid(_data) == false) {
      std::stringstream ss;
      ss << "current value is not valid: " << prop_traits<T>::serialize(_data);

      throw std::invalid_argument(ss.str().c_str());
    }
    force_set(_data);
  }

  void force_set(value_type _data) {
    data = _data;
  }

  /**
   * @copydoc PropertyHolder::setProp(const std::string &val)
   */
  void setProp(const std::string &val) {
    value_type value = prop_traits<T>::deserialize(val);
    set(value);
  }

  /**
   * @copydoc PropertyHolder::getProp()
   */
  std::string getProp() { return prop_traits<T>::serialize(data); }

private:
  value_type data; /**< Actual data */
};

/**
 * @brief Multiple property handler.
 */
template <typename... _PropTypes> struct Properties {
  typedef std::tuple<_PropertyHolder<_PropTypes>...> TupleType;
  typedef std::map<std::string, PropertyHolder *> MapType;

  /**
   * @brief contructor of properties. inits properties to default value and @a
   * type_map
   */
  Properties() { init_properties(items); }

  /**
   * @brief Property getter that uses type directly. It is recommened to use
   * this instead of string verion of getter
   * @code LayerType foo = std::get<LayerType>();
   */
  template <typename _PropType> _PropType get() {
    return std::get<_PropertyHolder<_PropType>>(items).get();
  }

  /**
   * @brief Property setter that uses type directly. It is recommened to use
   * this instead of string verion of setter
   * @throw exception::not_supported if property is not supported.
   * @code std::set<Beta1>(1.0);
   */
  template <typename _PropType>
  void set(typename prop_traits<_PropType>::value_type &&value) {
    std::get<_PropertyHolder<_PropType>>(items).set(
      std::forward<prop_traits<_PropType>>(value));
  }

  /**
   * @brief String version of property getter
   * @param[in] key key to find. from prop_traits<Type>::getKey()
   * @throw exception::not_supported if there are no key in the type_map
   */
  std::string get(const std::string &key) {
    MapType::iterator lb = type_map.find(key);

    if (lb != type_map.end()) {
      std::stringstream ss;
      ss << "key is not valid for current property type: " << key;
      throw exception::not_supported(ss.str());
    }

    return lb->second->getProp();
  }

  /**
   * @brief String version of property getter
   * @param[in] key key to find. from prop_traits<Type>::getKey()
   * @param[in] val value to set will eventually throw @a std::invalid_argument
   * if @a val is not valid.
   * @throw exception::not_supported if key is not valid.
   */
  void set(const std::string &key, const std::string &val) {
    MapType::iterator lb = type_map.lower_bound(key);

    if (lb == type_map.end()) {
      std::stringstream ss;
      ss << "key is not valid for current property type: " << key;
      throw exception::not_supported(ss.str());
    }

    lb->second->setProp(val);
  }

  /**
   * @brief load fle from vector of string
   * @param[in] v vector reprenstaion of key, value as {"key = value", "key2 =
   * value2"}
   */
  void load(const std::vector<std::string> &v) {
    std::string key, val;
    for (auto &i : v) {
      // getKeyVal(i, key, val);
      /**< NYI */
      // set(key, val)
    }
  }

  /**
   * @brief load fle from string bar reprentation
   * @param[in] bar_repr representation that takes like "key = value | key2 =
   * value2"
   */
  void load(const std::string &bar_repr) { /**< NYI */
  }

  /**
   * @brief load Ini Model file to property.
   * @param[in] path of the ini file
   */
  void loadIni(const std::string &path) { /**< NYI */
  }

  /**
   * @brief export current property to Ini
   * @param[in] path of the ini file
   */
  void saveIni(const std::string &path){/**< NYI */};

private:
  MapType type_map; /**< Helper map to get/set in string representation */
  TupleType items;  /**< Actual property holder */

  /// base case when tuple has reached the last (this is erroneuous case)
  template <std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  init_properties(std::tuple<Tp...> &t) {
    return;
  }

  /// general case
  template <std::size_t I = 0, typename... Tp>
    inline typename std::enable_if_t <
    I<sizeof...(Tp), void> init_properties(std::tuple<Tp...> &t) {

    using PropHolderType = std::decay_t<decltype(std::get<I>(t))>;

    using PropType = typename PropHolderType::prop_type;

    std::string key = prop_traits<PropType>::getKey();
    PropertyHolder* item = &std::get<I>(t);

    /// setting default value, this bypasses validation on purpose.
    std::get<I>(t).force_set(prop_traits<PropType>::getDefault());

    auto ret = type_map.insert(MapType::value_type(key, item));

    if (ret.second == false) {
      throw std::runtime_error("Duplicated entry for prop types.");
    }

    return init_properties<I + 1, Tp...>(t);
  }
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __POOLING_LAYER_H__ */
