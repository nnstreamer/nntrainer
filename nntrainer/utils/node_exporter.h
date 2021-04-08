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
  void save_result(std::tuple<Ts...> &props, ExportMethods method) {
    if (is_exported) {
      throw std::invalid_argument("This exporter is already used");
    }
    /** NYI!! */

    is_exported = true;
  }

  /**
   * @brief Get the result object
   *
   * @tparam methods method to get
   * @tparam T appropriate return type regarding the export method
   * @return T T
   */
  template <ExportMethods methods, typename T = return_type<methods>>
  T get_result() {
    if (!is_exported) {
      throw std::invalid_argument("This exporter is not exported anything yet");
    }
    /** NYI!! */
  }

private:
  bool is_exported;
};

} // namespace nntrainer
#endif // __NODE_EXPORTER_H__
