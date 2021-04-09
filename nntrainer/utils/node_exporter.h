// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file node_exporter.h
 * @date 08 April 2021
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

#ifndef __NODE_EXPORTER_H__
#define __NODE_EXPORTER_H__

namespace nntrainer {
enum class ExportMethods {
  METHOD_STRINGVECTOR = 0, /**< export to a string vector */
  METHOD_TFLITE = 1,       /**< epxort to tflite */
  METHOD_UNDEFINED = 999,  /**< undefined */
};

namespace {

template <ExportMethods method> struct return_type { using type = void; };

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

  std::vector<std::pair<std::string, std::string>>
    stored_result; /**< stored result */

  /// consider changing this to a promise / future if there is a async function
  /// involved to `save_result`
  bool is_exported; /**< boolean to check if exported */
};

} // namespace nntrainer
#endif // __NODE_EXPORTER_H__
