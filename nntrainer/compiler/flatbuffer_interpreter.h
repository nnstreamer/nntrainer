// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
 *
 * @file flatbuffer_interpreter.h
 * @date 09 February 2023
 * @brief NNTrainer flatbuffer Interpreter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __FLATBUFFER_INTERPRETER_H__
#define __FLATBUFFER_INTERPRETER_H__

#include <app_context.h>
#include <interpreter.h>

namespace nntrainer {

/**
 * @brief flatbuffer graph interpreter class
 *
 */
class FlatBufferInterpreter : public GraphInterpreter {
public:
  /**
   * @brief Construct a new flatbuffer Graph Interpreter object
   *
   * @param app_context_ app context to create layers
   */
  FlatBufferInterpreter(AppContext &app_context_ = AppContext::Global()) :
    app_context(app_context_) {}

  /**
   * @brief Destroy the flatbuffer Interpreter object
   *
   */
  virtual ~FlatBufferInterpreter() = default;

  /**
   * @copydoc GraphInterpreter::serialize(const std::string &out)
   */
  void serialize(const GraphRepresentation &representation,
                 const std::string &out) override;

  /**
   * @copydoc GraphInterpreter::deserialize(const std::string &in)
   */
  GraphRepresentation deserialize(const std::string &in) override;

private:
  AppContext &app_context;
};

} // namespace nntrainer

#endif // __FLATBUFFER_INTERPRETER_H__
