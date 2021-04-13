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

#include <fstream>
#include <string>

#include <tf_schema_generated.h>

#include <nntrainer_error.h>

static constexpr const char *FUNC_TAG = "[TFLITE INTERPRETER] ";

namespace {
/**
 * @brief after finishing building, call this to safe to a file
 *
 * @param builder flatbuffer builder
 * @param out out
 */
void builder2file(const flatbuffers::FlatBufferBuilder &builder,
                  const std::string &out) {
  uint8_t *buf = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  flatbuffers::Verifier v(buf, size);

  NNTR_THROW_IF(!tflite::VerifyModelBuffer(v), std::invalid_argument)
    << FUNC_TAG << "Verifying serialized model failed";

  std::ofstream os(out, std::ios_base::binary);
  NNTR_THROW_IF(!os.good(), std::invalid_argument)
    << FUNC_TAG << "failed to open, reason: " << strerror(errno);
  os.write((char *)builder.GetBufferPointer(), builder.GetSize());
  os.close();
}
} // namespace

namespace nntrainer {

void TfliteInterpreter::serialize(
  std::shared_ptr<const GraphRepresentation> representation,
  const std::string &out) {
  /// @todo check if graph is finalized
  flatbuffers::FlatBufferBuilder fb_builder;

  /// @todo parse subgraph
  /// 1. in&outs&weights
  /// 2. buffer
  /// 2. ops
  /// 3. op_codes

  auto desc = fb_builder.CreateString("This file is generated from NNTrainer");

  tflite::ModelBuilder model_builder(fb_builder);
  model_builder.add_version(3);
  model_builder.add_description(desc);
  auto model = model_builder.Finish();

  fb_builder.Finish(model, tflite::ModelIdentifier());

  builder2file(fb_builder, out);
}

std::shared_ptr<GraphRepresentation>
TfliteInterpreter::deserialize(const std::string &in) { /** NYI! */
  return nullptr;
}

} // namespace nntrainer
