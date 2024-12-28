// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file tflite_interpreter.h
 * @date 12 April 2021
 * @brief NNTrainer *.tflite Interpreter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __TFLITE_INTERPRETER_H__
#define __TFLITE_INTERPRETER_H__

#include <interpreter.h>

#include <engine.h>
namespace nntrainer {

/**
 * @brief tflite graph interpreter class
 *
 */
class TfliteInterpreter : public GraphInterpreter {
public:
  /**
   * @brief Construct a new tflite Graph Interpreter object
   *
   * @param app_context_ app context to create layers
   */
  TfliteInterpreter(const Engine &ct_engine_ = Engine::Global()) :
    ct_engine(ct_engine_) {}

  /**
   * @brief Destroy the Tflite Interpreter object
   *
   */
  virtual ~TfliteInterpreter() = default;

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
  Engine ct_engine;
};

} // namespace nntrainer

#endif // __TFLITE_INTERPRETER_H__
