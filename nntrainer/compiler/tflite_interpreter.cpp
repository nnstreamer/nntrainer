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
#include <tuple>
#include <type_traits>
#include <utility>

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
 * @brief Bidirectional Index map
 *
 * @tparam Key type of a underlying hashable value, please note that T will be
 * copied, so please use this for pointers and primitive values that is okay to
 * copy
 * @tparam Data data type to be stored inside the vector, if not given, same as
 * KeyType
 */
template <typename KeyType, typename DataType = KeyType>
class BidirectionalIndexMap {
public:
  /**
   * @brief addDatapoint to the map
   *
   * @param key key to be added to search for the data
   * @param data data to be added if there is no occurrence, data will be
   * copied.
   */
  void addDataWhenNotFound(KeyType key, DataType data) {
    auto search = key2index.find(key);

    if (search == key2index.end()) {
      key2index[key] = index2data.size();
      index2data.push_back(data);
    }
  }

  /**
   * @brief addDatapoint to the map when key and datatype is same
   *
   * @param key key/data to add
   */
  void addDataWhenNotFound(KeyType key) {
    static_assert(std::is_same<KeyType, DataType>::value == true,
                  "key type and data type are different!");
    addDataWhenNotFound(key, key);
  }

  /**
   * @brief Get the Index of the data
   *
   * @param key data that will be the key
   * @return unsigned int index
   */
  unsigned int getIndex(const KeyType &key) const {
    auto search = key2index.find(key);

    NNTR_THROW_IF(search == key2index.end(), std::invalid_argument)
      << FUNC_TAG << "Cannot find index for key: " << key;

    return search->second;
  }

  /**
   * @brief Get the Data object
   *
   * @param idx index to be searched
   * @return T datapoint T
   */
  DataType getData(unsigned int index) const {
    NNTR_THROW_IF(index >= index2data.size(), std::invalid_argument)
      << FUNC_TAG << "Cannot find data for index: " << index;

    return index2data[index];
  }

  /**
   * @brief Get the Data object
   *
   * @return const std::vector<T>& underlying data
   */
  const std::vector<DataType> &getData() const { return index2data; }

private:
  std::unordered_map<KeyType, unsigned int> key2index; /**< key -> index map */
  std::vector<DataType> index2data;                    /**< index -> data map */
};

/**
 * @brief tensorflow operation index map, this class manages operation index
 * mapping
 *
 */
class TfOpIdxMap {
public:
  using Buffer = std::pair<size_t, const float *>;

  TfOpIdxMap(const TfOpNodes &nodes) {
    auto &opcode_map = getIndexMap<tflite::BuiltinOperator>();
    auto update_opcode = [&opcode_map](tflite::BuiltinOperator opcode) {
      opcode_map.addDataWhenNotFound(opcode);
    };

    auto &buffer_map = getIndexMap<const float *, Buffer>();
    buffer_map.addDataWhenNotFound(
      nullptr, {0, empty_buffer}); // this represents undefined buffer
    buffer_map.addDataWhenNotFound(
      empty_buffer, {0, empty_buffer}); /// this represents empty buffer

    auto update_buffers = [&buffer_map](const TfOpNode::Variables &variables) {
      for (auto &variable : variables) {
        const Tensor &t = variable->getVariableRef();

        NNTR_THROW_IF(t.uninitialized() || !t.isAllocated(),
                      std::invalid_argument)
          << FUNC_TAG << "Weight tensor must be allocated";

        const float *buf = t.getData();
        buffer_map.addDataWhenNotFound(buf, {t.getSize(), buf});
      }
    };

    auto &variable_map = getIndexMap<const Var_Grad *>();
    auto update_variables =
      [&variable_map](const TfOpNode::Variables &variables) {
        for (auto &variable : variables) {
          variable_map.addDataWhenNotFound(variable);
        }
      };

    for (auto &op_node : nodes) {
      update_opcode(op_node.getOpType());
      update_variables(op_node.getInputs());
      update_variables(op_node.getOutputs());
      update_variables(op_node.getWeights());
      update_buffers(op_node.getWeights());
    }
  }

  template <typename KeyType, typename DataType = KeyType>
  BidirectionalIndexMap<KeyType, DataType> &getIndexMap() {
    return std::get<BidirectionalIndexMap<KeyType, DataType>>(maps);
  }

  template <typename KeyType, typename DataType = KeyType>
  const BidirectionalIndexMap<KeyType, DataType> &getIndexMap() const {
    return std::get<BidirectionalIndexMap<KeyType, DataType>>(maps);
  }

  const float *get_empty_buffer() const { return empty_buffer; }

private:
  float
    empty_buffer[0]; /**< reserved unintialized tensor points to this buffer */

  std::tuple<BidirectionalIndexMap<const float *, Buffer>,   /**< buffer map */
             BidirectionalIndexMap<tflite::BuiltinOperator>, /**< opcode map */
             BidirectionalIndexMap<const Var_Grad *>>        /**< tensor map */
    maps;
};

TfOpNodes
buildOpNodes(std::shared_ptr<const GraphRepresentation> representation) {
  TfOpNodes nodes;
  /// @todo, look ahead of layers to get nodes that can be fused
  /// we will need to have a dedicated builder
  for (auto iter = representation->cbegin(); iter != representation->cend();
       iter++) {
    const auto &ln = *iter;
    nodes.emplace_back(*ln->getObject());
  }

  return nodes;
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Buffer>>>
buildBuffers(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  const auto &buffers =
    map.getIndexMap<const float *, TfOpIdxMap::Buffer>().getData();

  std::vector<flatbuffers::Offset<tflite::Buffer>> fb_buffers;
  fb_buffers.reserve(buffers.size());

  auto create_buffer_offset = [&fbb](const TfOpIdxMap::Buffer &buffer) {
    if (buffer.first == 0) {
      return tflite::CreateBuffer(fbb);
    }

    auto data = fbb.CreateVector(
      reinterpret_cast<const uint8_t *>(buffer.second), buffer.first);

    return tflite::CreateBuffer(fbb, data);
  };

  std::transform(buffers.begin(), buffers.end(), std::back_inserter(fb_buffers),
                 create_buffer_offset);
  return fbb.CreateVector(fb_buffers);
}

flatbuffers::Offset<
  flatbuffers::Vector<flatbuffers::Offset<tflite::OperatorCode>>>
buildOperatorCodes(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  const auto &op_codes = map.getIndexMap<tflite::BuiltinOperator>().getData();

  std::vector<flatbuffers::Offset<tflite::OperatorCode>> fb_op_codes;
  fb_op_codes.reserve(op_codes.size());

  auto create_op_offset = [&fbb](const tflite::BuiltinOperator &op,
                                 int32_t version = 1) {
    tflite::OperatorCodeBuilder builder(fbb);
    builder.add_deprecated_builtin_code(static_cast<int8_t>(op));
    /// @todo find reason why version field is not shown
    /// on json when version is 1 (other versions are fine)
    builder.add_version(version);
    builder.add_builtin_code(op);
    return builder.Finish();
  };

  std::transform(op_codes.begin(), op_codes.end(),
                 std::back_inserter(fb_op_codes), create_op_offset);

  return fbb.CreateVector(fb_op_codes);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Tensor>>>
buildTensors(const TfOpIdxMap &map, flatbuffers::FlatBufferBuilder &fbb) {
  /// @todo: the actual (suqeezed) tensor dimension must be known before coming
  /// here. For now, it is directly guessed for the fc layer
  const auto &variables = map.getIndexMap<const Var_Grad *>().getData();
  const auto &buffer_map = map.getIndexMap<const float *, TfOpIdxMap::Buffer>();

  std::vector<flatbuffers::Offset<tflite::Tensor>> fb_tensors;
  fb_tensors.reserve(variables.size());

  auto create_tensor = [&fbb, &buffer_map](const Var_Grad *var) {
    bool need_shape_signature = var->getDim().is_dynamic();
    std::vector<int32_t> eff_dim = var->getDim().getEffectiveDimension();
    auto shape = fbb.CreateVector(eff_dim);

    decltype(shape) shape_sig;
    if (need_shape_signature) {
      std::vector<int32_t> dyn_dim = var->getDim().getEffectiveDimension(true);
      shape_sig = fbb.CreateVector(dyn_dim);
    }

    auto name = fbb.CreateString(var->getName());
    auto tensor = var->getVariableRef();

    unsigned int buffer_idx = 1;
    if (!tensor.uninitialized() && tensor.isAllocated()) {
      buffer_idx = buffer_map.getIndex(var->getVariableRef().getData());
    }

    tflite::TensorBuilder builder(fbb);
    builder.add_name(name);
    builder.add_buffer(buffer_idx);
    builder.add_type(tflite::TensorType_FLOAT32);
    builder.add_shape(shape);
    if (need_shape_signature) {
      builder.add_shape_signature(shape_sig);
    }
    return builder.Finish();
  };

  std::transform(variables.begin(), variables.end(),
                 std::back_inserter(fb_tensors), create_tensor);

  return fbb.CreateVector(fb_tensors);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>>
buildSubGraphs(const TfOpNodes &nodes, const TfOpIdxMap &map,
               flatbuffers::FlatBufferBuilder &fbb) {

  auto tensors = buildTensors(map, fbb);

  /// @todo extract this to buildSubgraph if there is one or more subgraph
  auto name = fbb.CreateString("main");

  auto builder = tflite::SubGraphBuilder(fbb);
  builder.add_tensors(tensors);
  builder.add_name(name);
  auto subgraph = builder.Finish();

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  subgraphs.reserve(1);
  subgraphs.push_back(subgraph);

  return fbb.CreateVector(subgraphs);
}

} // namespace

void TfliteInterpreter::serialize(
  std::shared_ptr<const GraphRepresentation> representation,
  const std::string &out) {
  /// @todo check if graph is finalized & initialized and ready to serialize.
  /// 1. The graph must have weights, input dims, output dims set
  flatbuffers::FlatBufferBuilder fbb;

  auto opNodes = buildOpNodes(representation);
  TfOpIdxMap map(opNodes); /// build TfOpIdxMap from opNodes

  auto opcodes = buildOperatorCodes(map, fbb);
  auto subgraphs = buildSubGraphs(opNodes, map, fbb);
  auto buffers = buildBuffers(map, fbb);
  auto desc = fbb.CreateString("This file is generated from NNTrainer");

  tflite::ModelBuilder model_builder(fbb);

  model_builder.add_operator_codes(opcodes);
  model_builder.add_subgraphs(subgraphs);
  model_builder.add_buffers(buffers);
  model_builder.add_version(3);
  model_builder.add_description(desc);
  auto model = model_builder.Finish();

  fbb.Finish(model, tflite::ModelIdentifier());
  builder2file(fbb, out);
}

std::shared_ptr<GraphRepresentation>
TfliteInterpreter::deserialize(const std::string &in) {
  /// ======== list of things to consider ========
  /// we need to reconstruct some properties from the shape
  /// eg) units are not saved as a property

  /** NYI! */
  return nullptr;
}

} // namespace nntrainer
