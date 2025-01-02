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

#ifndef __INI_INTERPRETER_H__
#define __INI_INTERPRETER_H__

#include <memory>
#include <string>

#include <iniparser.h>

#include <engine.h>
#include <interpreter.h>

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
    const Engine &ct_engine_ = Engine::Global(),
    std::function<const std::string(const std::string &)> pathResolver_ =
      [](const std::string &path) { return path; });

  /**
   * @brief Destroy the Ini Graph Interpreter object
   *
   */
  virtual ~IniGraphInterpreter();

  /**
   * @copydoc GraphInterpreter::serialize(const GraphRepresentation
   * representation, const std::string &out)
   */
  void serialize(const GraphRepresentation &representation,
                 const std::string &out) override;

  /**
   * @copydoc GraphInterpreter::deserialize(const std::string &in)
   */
  GraphRepresentation deserialize(const std::string &in) override;

private:
  Engine ct_engine;
  std::function<const std::string(std::string)> pathResolver;
};

} // namespace nntrainer

#endif // __INI_INTERPRETER_H__
