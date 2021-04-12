// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file node_exporter.h
 * @date 09 April 2021
 * @brief NNTrainer Node exporter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <nntrainer_error.h>
#include <parse_util.h>

#ifndef __NODE_EXPORTER_H__
#define __NODE_EXPORTER_H__

namespace nntrainer {

/**
 * @brief Defines Export Method to be called with
 *
 */
enum class ExportMethods {
  METHOD_STRINGVECTOR = 0, /**< export to a string vector */
  METHOD_TFLITE = 1,       /**< epxort to tflite */
  METHOD_UNDEFINED = 999,  /**< undefined */
};

namespace {

/**
 * @brief meta function that return return_type when a method is being called
 *
 * @tparam method returned when certain method is being called
 */
template <ExportMethods method> struct return_type { using type = void; };

/**
 * @brief meta function to check return type when the method is string vector
 *
 * @tparam specialized so not given
 */
template <> struct return_type<ExportMethods::METHOD_STRINGVECTOR> {
  using type = std::vector<std::pair<std::string, std::string>>;
};
} // namespace

/**
 * @brief Exporter class helps to exports the node information in a predefined
 * way. because each method will require complete different methods, this class
 * exploits visitor pattern to make a custom defined saving method
 *
 */
class Exporter {
public:
  /**
   * @brief Construct a new Exporter object
   *
   */
  Exporter() : is_exported(false){};

  /**
   * @brief this function iterates over the property and process the property in
   * a designated way.
   *
   * @tparam Ts type of elements
   * @param props tuple that contains properties
   * @param method method to export
   */
  template <typename... Ts>
  void save_result(const std::tuple<Ts...> &props, ExportMethods method) {
    switch (method) {

    case ExportMethods::METHOD_STRINGVECTOR: {

      /**
       * @brief function to pass to the iterate_prop, this saves the property to
       * stored_result
       *
       * @param prop property property to pass
       * @param index index of the current property
       */
      auto callable = [this](auto &&prop, size_t index) {
        std::string key = std::remove_reference_t<decltype(prop)>::key;
        stored_result.emplace_back(key, to_string(prop));
      };
      iterate_prop(callable, props);
    } break;
    case ExportMethods::METHOD_TFLITE:
    /// fall thorugh intended (NYI!!)
    case ExportMethods::METHOD_UNDEFINED:
    /// fall thorugh intended
    default:
      throw exception::not_supported("given method is not supported yet");
    }

    is_exported = true;
  }

  /**
   * @brief Get the result object
   *
   * @tparam methods method to get
   * @tparam T appropriate return type regarding the export method
   * @return T T
   */
  template <ExportMethods methods,
            typename T = typename return_type<methods>::type>
  const T &get_result();

private:
  std::vector<std::pair<std::string, std::string>>
    stored_result; /**< stored result */

  /// consider changing this to a promise / future if there is a async function
  /// involved to `save_result`
  bool is_exported; /**< boolean to check if exported */
};

/**
 * @brief base case of iterate_prop, iterate_prop iterates the given tuple
 *
 * @tparam I size of tuple(automated)
 * @tparam Callable generic lambda to be called during iteration
 * @tparam Ts types from tuple
 * @param c callable gerneric labmda
 * @param tup tuple to be iterated
 * @return void
 */
template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<I == sizeof...(Ts), void>::type
iterate_prop(Callable &&c, const std::tuple<Ts...> &tup) {
  // end of recursion;
}

/**
 * @brief base case of iterate_prop, iterate_prop iterates the given tuple
 *
 * @tparam I size of tuple(automated)
 * @tparam Callable generic lambda to be called during iteration
 * @tparam Ts types from tuple
 * @param c callable gerneric labmda
 * @param tup tuple to be iterated
 * @return not used
 */
template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type
iterate_prop(Callable &&c, const std::tuple<Ts...> &tup) {
  c(std::get<I>(tup), I);

  iterate_prop<I + 1>(c, tup);
}

/**
 * @copydoc  template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type iterate_prop(Callable
&&c, const std::tuple<Ts...> &tup)
 */
template <size_t I = 0, typename Callable, typename... Ts>
typename std::enable_if<(I < sizeof...(Ts)), void>::type
iterate_prop(Callable &&c, std::tuple<Ts...> &tup) {
  c(std::get<I>(tup), I);

  iterate_prop<I + 1>(c, tup);
}

/**
 * @brief load property from the api formatted string ({"key=value",
 * "key1=value1"})
 *
 * @tparam Ts prop type
 * @param string_vector api formatted string;
 * @param[out] props props to be iterated
 * @return std::vector<std::string> vector of string that is not used while
 * setting the property
 */
template <typename... Ts>
std::vector<std::string>
load_properties(const std::vector<std::string> &string_vector,
                std::tuple<Ts...> &props) {

  std::vector<std::pair<std::string, std::string>> left;
  left.reserve(string_vector.size());

  std::transform(string_vector.begin(), string_vector.end(),
                 std::back_inserter(left), [](const std::string &property) {
                   std::string key, value;
                   int status = getKeyValue(property, key, value);
                   NNTR_THROW_IF(status != ML_ERROR_NONE, std::invalid_argument)
                     << "parsing property failed, original format: "
                     << property;
                   return std::make_pair(key, value);
                 });

  auto callable = [&left](auto &&prop, size_t index) {
    std::string prop_key = std::remove_reference_t<decltype(prop)>::key;

    for (auto iter = left.begin(); iter < left.end(); ++iter) {
      if (istrequal(prop_key, iter->first) == true) {
        from_string(iter->second, prop);
        iter = left.erase(iter);
      }
    }
  };

  iterate_prop(callable, props);

  std::vector<std::string> remainder;
  remainder.reserve(left.size());

  std::transform(left.begin(), left.end(), std::back_inserter(remainder),
                 [](const decltype(left)::value_type &v) {
                   return v.first + "=" + v.second;
                 });

  return remainder;
}

} // namespace nntrainer
#endif // __NODE_EXPORTER_H__
