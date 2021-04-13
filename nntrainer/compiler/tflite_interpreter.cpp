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

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

#include <tf_schema_generated.h>

#include <fc_layer.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>
#include <var_grad.h>
#include <weight.h>

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
public:
  using Variables = std::vector<const Var_Grad *>;

  TfOpNode() = default;

  /**
   * @brief Construct a new Tf Op Node object from layer
   * @note this is a shortcut to skip if layer does not need to be devided or
   * fused
   * @param layer layer that is converted to TfOpNode
   */
  TfOpNode(const Layer &layer) {
    setInputs(layer.getInputRef());
    setOutputs(layer.getOutputRef());
    setWeights(layer.getWeightsRef());
    setOpType(layer.getType());
  }

  /**
   * @brief Set the Inputs object from layer
   *
   * @param inputs_ input to be inserted
   */
  void setInputs(const std::vector<std::shared_ptr<Var_Grad>> &inputs_) {
    inputs.reserve(inputs_.size());
    std::transform(inputs_.begin(), inputs_.end(), std::back_inserter(inputs),
                   [](const auto &data) { return data.get(); });
  }

  /**
   * @brief Set the Outputs object
   *
   * @param outputs_ output to be inserted
   */
  void setOutputs(const std::vector<std::shared_ptr<Var_Grad>> &outputs_) {
    outputs.reserve(outputs_.size());
    std::transform(outputs_.begin(), outputs_.end(),
                   std::back_inserter(outputs),
                   [](const auto &data) { return data.get(); });
  }

  /**
   * @brief Set the Weights object
   *
   * @param weights_ set weights from the object
   */
  void setWeights(const std::vector<Weight> &weights_) {
    weights.reserve(weights_.size());
    std::transform(weights_.begin(), weights_.end(),
                   std::back_inserter(weights),
                   [](const auto &data) { return &data; });
  }

  /**
   * @brief Set the Op Type object
   * @todo Considering number of alternatives to optimize this, for now it is
   * just workable.
   * 1. add and maintain global unordered map
   * 2. Save information in the appcontext later we can retrieve
   * 3. let type be an immutable property and let exporter handle this instead
   * of this method (preferrable)
   * @param type type to convert
   */
  void setOpType(const std::string &type) {
    if (istrequal(type, FullyConnectedLayer::type)) {
      setOpType(tflite::BuiltinOperator_FULLY_CONNECTED);
      return;
    }

    throw std::invalid_argument("not supported type");
  }

  /**
   * @brief Set the Builtin Options object,
   * @note this can go private, export from a layer and fill this out
   *
   * @param builtin_option_type_ builtin option type
   * @param builtin_ops_ flatbuffer offset of builtin_ops
   */
  void setBuiltinOptions(tflite::BuiltinOptions builtin_option_type_,
                         flatbuffers::Offset<void> &builtin_ops_) {
    builtin_ops = builtin_ops_;
    builtin_option_type = builtin_option_type_;
  }

  /**
   * @brief Get the Inputs object
   *
   * @return Variables& inputs
   */
  Variables &getInputs() { return inputs; }
  const Variables &getInputs() const { return inputs; }

  /**
   * @brief Get the Outputs object
   *
   * @return Variables&
   */
  Variables &getOutputs() { return outputs; }
  const Variables &getOutputs() const { return outputs; }

  /**
   * @brief Get the Weights object
   *
   * @return Variables&
   */
  Variables &getWeights() { return weights; }
  const Variables &getWeights() const { return weights; }

  /**
   * @brief Get the Op Type object
   *
   * @return const tflite::BuiltinOperator
   */
  const tflite::BuiltinOperator getOpType() const { return op_type; }

private:
  /**
   * @brief Set the Op Type object
   *
   * @param op_type_ operation type
   */
  void setOpType(tflite::BuiltinOperator op_type_) { op_type = op_type_; }

  Variables inputs;  /**< input variables */
  Variables outputs; /**< output variables */
  Variables weights; /**< weight variables */

  tflite::BuiltinOperator op_type;

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
  TfOpNodes nodes;
  /// @todo, look ahead of layers to get nodes that can be fused
  for (const auto &ln : representation->getSorted()) {
    nodes.emplace_back(*ln->getObject());
    std::cout << ln->getObject()->getName() << '\n';
  }

  return nodes;
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
