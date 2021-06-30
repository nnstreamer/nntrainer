// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file ini_interpreter.h
 * @date 02 April 2021
 * @brief NNTrainer Ini Interpreter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <iostream>
#include <memory>
#include <string>

#include <iniparser.h>

#include <app_context.h>
#include <interpreter.h>

#ifndef __INI_INTERPRETER_H__
#define __INI_INTERPRETER_H__

namespace nntrainer {

/**
 * @brief ini graph interpreter class
 *
 */
class IniGraphInterpreter : public GraphInterpreter {
public:
  /**
   * @brief Construct a new Ini Graph Interpreter object
   *
   * @param pathResolver_ path resolver function to be used
   */
  IniGraphInterpreter(
    const AppContext &app_context_ = AppContext::Global(),
    std::function<const std::string(const std::string &)> pathResolver_ =
      [](const std::string &path) { return path; }) :
    app_context(app_context_),
    pathResolver(pathResolver_) {}

  /**
   * @brief Destroy the Ini Graph Interpreter object
   *
   */
  virtual ~IniGraphInterpreter(){};

  /**
   * @copydoc GraphInterpreter::serialize(const std::string &out)
   */
  void serialize(std::shared_ptr<const GraphRepresentation> representation,
                 const std::string &out) override;

  /**
   * @copydoc GraphInterpreter::deserialize(const std::string &in)
   */
  std::shared_ptr<GraphRepresentation>
  deserialize(const std::string &in) override;

private:
  /**
   * @brief Create a Layer From Section object
   *
   * @param ini ini if throw, ini will be freed.
   * @param section section name
   * @return std::shared_ptr<Layer> layer
   */
  std::shared_ptr<Layer> loadLayerConfig(dictionary *ini,
                                         const std::string &section);

  /**
   * @brief Create a Layer From Backbone Config
   *
   * @param ini ini if throw, ini will be freed.
   * @param section section name
   * @return std::shared_ptr<Layer> layer
   */
  std::shared_ptr<Layer> loadBackboneConfigIni(dictionary *ini,
                                               const std::string &section);

  AppContext app_context;
  std::function<const std::string(std::string)> pathResolver;
};

} // namespace nntrainer

#endif // __INI_INTERPRETER_H__
