// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <pk.kapoor@samsung.com>
 *
 * @file   module.h
 * @date   13 October 2021
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is module interface for c++ API. A module contains a pack of
 * layers, interpreted in a defined way.
 *
 * @note  This is experimental API and not stable.
 */
#ifndef __MODULE_H__
#define __MODULE_H__

#include <layer.h>
#include <memory>

#ifdef __cplusplus
#if __cplusplus >= MIN_CPP_VERSION

namespace ml {
namespace train {

/**
 * @brief     Enumeration of Module Type
 */
enum class ModuleType {
  NORMAL,    /** Normal graph wihtout any modification */
  RECURRENT, /** Recurrent module */
  UNKNOWN    /** Unknown */
};

/**
 * @brief Module Interface to generate a sequence of graph way.
 *
 */
class Module {
public:
  /**
   * @brief Destroy the Module object
   *
   */
  virtual ~Module() {}

  /**
   * @brief     set Property of a module
   * @param     values values of property
   * @details   This function accepts vector of properties in the format -
   *  { std::string "property_name=property_value", ...}
   */
  virtual void setProperty(const std::vector<std::string> &properties) = 0;

  /**
   * @brief     add layer into the module. Layer must be created
   * @param     layer to pass ownership
   */
  virtual void addLayer(std::unique_ptr<Layer> &&layer) = 0;

  /**
   * @brief finalize the module and mark it as ready to use
   *
   */
  void finalize();

  /**
   * @brief Render the module into vector of nodes to be added to a model.
   * @note Calling multiple times should result in the same graph topology with
   * the same configuration but having different actual node to prevent
   * unintended node sharing.
   * @note Finalize must be called before calling this function.
   *
   * @param scope scope of the graph
   * @param input_layers name input layers this module should be
   * connected to
   * @return std::vector<std::shared_ptr<Layer>> list of processed nodes
   */
  virtual std::vector<std::shared_ptr<Layer>>
  render(const std::string &scope,
         const std::vector<std::string> &input_layers) = 0;
};

/**
 * @brief Create a Module object
 *
 * @param type type
 * @param properties properties
 * @return std::unique_ptr<Module> created modeul
 */
std::unique_ptr<Module>
createModule(ModuleType type, const std::vector<std::string> &properties = {});

} // namespace train
} // namespace ml

#endif // __cpluscplus
#endif // __cpluscplus
#endif // __MODULE_H__
