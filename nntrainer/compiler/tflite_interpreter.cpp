// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file tflite_interpreter.cpp
 * @date 12 April 2021
 * @brief NNTrainer *.tflite Interpreter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tflite_interpreter.h>

#include <tf_schema_generated.h>

namespace nntrainer {

void TfliteInterpreter::serialize(
  std::shared_ptr<const GraphRepresentation> representation,
  const std::string &out) {
  /** NYI!! */
}

std::shared_ptr<GraphRepresentation>
TfliteInterpreter::deserialize(const std::string &in) { /** NYI! */
  return nullptr;
}

} // namespace nntrainer
