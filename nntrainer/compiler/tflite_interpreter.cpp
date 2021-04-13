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
#include <memory>
#include <string>

#include <tf_schema_generated.h>

#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>

#define UNUSED(x) x __attribute__((unused))
#define WILL(x) ;

static constexpr const char *FUNC_TAG = "[TFLITE INTERPRETER] ";

namespace nntrainer {
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

/**
 * @brief tensorflow operational node representation. This class contains,
 * information to build operation flatbuffer
 *
 */
class TfOpNode {
private:
  std::vector<std::shared_ptr<Var_Grad>> inputs;
  std::vector<std::shared_ptr<Var_Grad>> outputs;
  std::vector<std::shared_ptr<Var_Grad>> weights;

  tflite::BuiltinOperator ops_type;

  /// retrieve this from export_to
  flatbuffers::Offset<void> builtin_ops;
  tflite::BuiltinOptions builtin_option_type;
};

using TfOpNodes = std::vector<TfOpNode>;

/**
 * @brief tensorflow operation index map, this class manages operation index
 * mapping
 *
 */
class TfOpIdxMap {
public:
  TfOpIdxMap(std::vector<TfOpNode> nodes){};

private:
  /**
   * @brief Bidirectional Index map
   *
   * @tparam T type of a underyling value
   */
  template <typename T> class BidirectionalIndexMap {
    std::unordered_map<T, unsigned int> data2index; /**< data -> index map */
    std::vector<T> index2data;                      /**< index -> data map */
  };

  float empty_buffer[0]; /**< unintialized tensor points to this buffer */

  BidirectionalIndexMap<float *> buffer_map; /**< underlying buffer map */
  BidirectionalIndexMap<tflite::BuiltinOperator> opcode_map; /**< opcode map */
  BidirectionalIndexMap<Var_Grad *> variable_map;            /**< tensor map */
};

TfOpNodes
buildOpNodes(std::shared_ptr<const GraphRepresentation> representation) {
  /** NYI! */
  return TfOpNodes();
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>>
buildBuffers(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  /** NYI! */
  return flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>>();
}

flatbuffers::Offset<
  flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>
buildOperatorCodes(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  /** NYI! */
  return flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>();
};

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>>
buildSubGraph(const TfOpNodes &nodes, const TfOpIdxMap &map,
              flatbuffers::FlatBufferBuilder &fbb) {
  /** NYI! */
  return flatbuffers::Offset<
    flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>>();
}

} // namespace

void TfliteInterpreter::serialize(
  std::shared_ptr<const GraphRepresentation> representation,
  const std::string &out) {
  /// @todo check if graph is finalized
  flatbuffers::FlatBufferBuilder fbb;

  auto opNodes = buildOpNodes(representation);
  TfOpIdxMap map(opNodes); /// build TfOpIdxMap

  auto UNUSED(opcodes) = buildOperatorCodes(map, fbb);
  auto UNUSED(buffers) = buildBuffers(map, fbb);
  auto UNUSED(subgraph) = buildSubGraph(opNodes, map, fbb);
  auto desc = fbb.CreateString("This file is generated from NNTrainer");

  tflite::ModelBuilder model_builder(fbb);

  WILL(model_builder.add_operator_codes(opcode_offset));
  WILL(model_builder.add_buffers(buffers));
  WILL(model_builder.add_subgraphs(subgraph));
  model_builder.add_version(3);
  model_builder.add_description(desc);
  auto model = model_builder.Finish();

  fbb.Finish(model, tflite::ModelIdentifier());
  builder2file(fbb, out);
}

std::shared_ptr<GraphRepresentation>
TfliteInterpreter::deserialize(const std::string &in) { /** NYI! */
  return nullptr;
}

} // namespace nntrainer
