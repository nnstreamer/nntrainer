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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include <tf_schema_generated.h>

#include <bn_realizer.h>
#include <fc_layer.h>
#include <layer_node.h>
#include <loss_realizer.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>
#include <tf_schema_generated.h>
#include <tflite_opnode.h>

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
 * @brief get predecessor nodes
 *
 * @param node the node from which to get predecessor nodes
 * @note virtual nodes are ignored
 */
std::vector<const TfOpNode *> getPredNodes(const TfOpNode &node) {
  std::vector<const TfOpNode *> predNodes;

  for (auto input : node.getInputNodes()) {
    const TfOpNode *pred = input;
    while (pred->isVirtualNode()) {
      /// Assume that virtual nodes have single input
      assert(pred->arity() == 1);
      pred = pred->arg(0);
    }
    predNodes.push_back(pred);
  }
  return predNodes;
}

using TfOpNodes = std::vector<std::unique_ptr<TfOpNode>>;

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

    auto update_buffer_map = [&buffer_map](const TfOpNode::Variables &variables,
                                           bool dynamic) {
      for (auto &variable : variables) {
        const float *buf = variable->getData();
        assert(buf != nullptr);
        auto byte_size = dynamic ? 0 : variable->bytes();
        buffer_map.addDataWhenNotFound(buf, {byte_size, buf});
      }
    };

    auto register_tensors =
      [&tensors = this->tensors](const TfOpNode::Variables &variables) {
        for (auto &variable : variables) {
          auto tensor_it = std::find(tensors.begin(), tensors.end(), variable);
          if (tensor_it == tensors.end()) {
            tensors.push_back(variable);
          }
        }
      };

    for (auto &op_node : nodes) {
      if (op_node->isVirtualNode())
        continue;
      update_opcode(op_node->getOpType());

      if (op_node->isInputNode()) {
        /**
         * Q) Why only register graph input tensor?
         *
         * A) the tflite needs only one tensor between nodes. Therefore,
         *basically, no inputs are considered except graph input that doesn't
         *have FROM node.
         **/
        register_tensors(op_node->getInputs());
        /**
         * Q) Why only update second input of the input node?
         *
         * A) 1. graph input nodes should be Transpose operator to change data
         *format from NCHW to NHWC.
         *    2. Transpose operator has two inputs - input to be
         *transposed(input[0]), 1d permute vector(input[1])
         *    3. input[0] has nullptr data pointer, which can't be added to
         *buffer_map. But, input[0] should have its own buffer and it will be
         *considered when the tflite buffers are built.
         **/
        assert(op_node->getInputs()[0]->getData() == nullptr);
        update_buffer_map({op_node->getInputs()[1]}, false);
      }
      register_tensors(op_node->getWeights());
      update_buffer_map(op_node->getWeights(), false);

      register_tensors(op_node->getOutputs());
      update_buffer_map(op_node->getOutputs(), true);
    }

    auto update_model_io_to = [this](const TfOpNode::Variables &variables,
                                     std::vector<int> &v) {
      for (auto &variable : variables) {
        if (variable->getName().find("nntrainer_internal_perm") !=
            std::string::npos)
          continue;
        v.push_back(this->getTensorIndex(variable));
      }
    };

    for (auto &op_node : nodes) {
      if (op_node->isVirtualNode())
        continue;
      if (op_node->isInputNode()) {
        update_model_io_to(op_node->getInputs(), inputs);
      }
      if (op_node->isOutputNode()) {
        update_model_io_to(op_node->getOutputs(), outputs);
      }
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

  const std::vector<int> &getInputs() const { return inputs; }

  const std::vector<int> &getOutputs() const { return outputs; }

  const std::vector<const Tensor *> &getTensors() const { return tensors; }

  std::ptrdiff_t getTensorIndex(const Tensor *tensor) const {
    auto tensor_it = std::find(tensors.begin(), tensors.end(), tensor);
    NNTR_THROW_IF(tensor_it == tensors.cend(), std::invalid_argument)
      << FUNC_TAG << "Cannot find index for tensor: " << tensor->getName();
    return std::distance(tensors.begin(), tensor_it);
  }

private:
  float empty_buffer[0]; /**< reserved unintialized tensor points to this
                            buffer */

  std::tuple<BidirectionalIndexMap<const float *, Buffer>,   /**< buffer map
                                                              */
             BidirectionalIndexMap<tflite::BuiltinOperator>> /**< opcode map
                                                              */
    maps;

  std::vector<int> inputs;
  std::vector<int> outputs;
  /// since it is used as a tensor index, the order is important
  std::vector<const Tensor *> tensors;
};

TfOpNodes buildOpNodes(const GraphRepresentation &representation,
                       flatbuffers::FlatBufferBuilder &fbb) {
  TfOpNodes nodes;
  /// @todo TfOpNode needs to have LayerNode pointer
  std::map<TfOpNode *, const LayerNode *> tf_to_layer;
  std::map<const LayerNode *, TfOpNode *> layer_to_tf;
  /// @todo, look ahead of layers to get nodes that can be fused
  /// we will need to have a dedicated builder
  for (auto iter = representation.cbegin(); iter != representation.cend();
       iter++) {
    const auto &ln = *iter;
    Exporter e(&fbb);
    ln->exportTo(e, ml::train::ExportMethods::METHOD_TFLITE);

    nodes.emplace_back(e.getResult<ml::train::ExportMethods::METHOD_TFLITE>());
    tf_to_layer.insert({nodes.back().get(), ln.get()});
    layer_to_tf.insert({ln.get(), nodes.back().get()});
  }

  /// set arity of TfOpNodes
  for (auto &n : nodes) {
    auto tf_node = n.get();
    auto layer_node = tf_to_layer.find(tf_node)->second;
    auto layer_node_inputs = layer_node->getInputConnections();

    /// assume that the TfOpNode and the LayerNode have a one-to-one
    /// relationship
    tf_node->arity(layer_node_inputs.size());
    for (size_t index = 0; index < layer_node_inputs.size(); index++) {
      auto input_layer_name = layer_node_inputs[index];
      auto input_layer_node =
        std::find_if(
          representation.begin(), representation.end(),
          [&input_layer_name](std::shared_ptr<nntrainer::LayerNode> node) {
            return istrequal(node.get()->getName(), input_layer_name);
          })
          ->get();
      tf_node->setArg(index, layer_to_tf.find(input_layer_node)->second);
    }
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

  // add input buffer
  for (unsigned index = 0; index < map.getInputs().size(); index++) {
    fb_buffers.push_back(create_buffer_offset({0, nullptr}));
  }
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
  /// @todo: the actual (suqeezed) tensor dimension must be known before
  /// coming here. For now, it is directly guessed for the fc layer
  const auto &variables = map.getTensors();
  const auto &buffer_map = map.getIndexMap<const float *, TfOpIdxMap::Buffer>();
  auto graph_input_offset = map.getInputs().size() - 1;

  std::vector<flatbuffers::Offset<tflite::Tensor>> fb_tensors;
  fb_tensors.reserve(variables.size());

  auto create_tensor = [&fbb, &buffer_map,
                        &graph_input_offset](const Tensor *var) {
    auto dim = var->getDim();
    bool need_shape_signature = dim.is_dynamic();
    std::vector<int32_t> eff_dim = dim.getEffectiveDimension();
    auto shape = fbb.CreateVector(eff_dim);

    decltype(shape) shape_sig;
    if (need_shape_signature) {
      std::vector<int32_t> dyn_dim = dim.getEffectiveDimension(true);
      shape_sig = fbb.CreateVector(dyn_dim);
    }

    /// change this var->getName when tensor have it's own name
    auto name = fbb.CreateString("nntrainer_converted" + var->getName());

    /// only graph inputs have nullptr data pointer.
    unsigned int buffer_idx =
      var->getData() == nullptr
        ? buffer_map.getData().size() - graph_input_offset--
        : buffer_map.getIndex(var->getData());

    tflite::TensorBuilder builder(fbb);
    builder.add_name(name);
    builder.add_buffer(buffer_idx);
    /// @todo support more data types
    /// @note this is workaround because nntrainer tensor allows only float
    /// dtype
    if (var->getName().find("nntrainer_internal_perm") != std::string::npos) {
      builder.add_type(tflite::TensorType_INT32);
    } else
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

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::Operator>>>
buildOperators(const TfOpNodes &nodes, const TfOpIdxMap &map,
               flatbuffers::FlatBufferBuilder &fbb) {

  /// this lambda maps variables to list of indexes in the map
  auto variables_to_idx_vector = [&map](const TfOpNode::Variables &v) {
    std::vector<int> idx_vector;
    idx_vector.reserve(v.size());

    std::transform(
      v.begin(), v.end(), std::back_inserter(idx_vector),
      [&map](const Tensor *variable) { return map.getTensorIndex(variable); });
    return idx_vector;
  };

  auto create_operator = [&fbb, &map,
                          &variables_to_idx_vector](const TfOpNode &node) {
    auto &index_map = map.getIndexMap<tflite::BuiltinOperator>();

    auto op_code = index_map.getIndex(node.getOpType());
    std::vector<int> inputs;
    if (node.isInputNode()) {
      inputs = variables_to_idx_vector(node.getInputs());
    } else {
      /**
       *  Q) Why find a tensor that shares a buffer with input tensor?
       *
       *  A) the tflite needs only one tensor between nodes. Therefore,
       *basically, output tensors are used for tflite tensor that shares its
       *buffer with input's
       **/
      TfOpNode::Variables input_tensors;
      for (auto parent_node : getPredNodes(node)) {
        for (auto parent_out : parent_node->getOutputs()) {
          for (auto in : node.getInputs()) {
            /// second condition is a workaround
            /// Transpose op output tensor originally had nullptr data pointer
            /// but it has been allocated (parent_out->getData()). But, the
            /// buffer that shared its buffer hasn't so it has still nullptr
            /// (in->getData()).
            /// @todo remove this workaround
            if (parent_out->getData() == in->getData() ||
                (in->getData() == nullptr && parent_out->getData())) {
              if (std::find(input_tensors.begin(), input_tensors.end(),
                            parent_out) != input_tensors.end())
                continue;
              input_tensors.push_back(parent_out);
            }
          }
        }
      }
      inputs = variables_to_idx_vector(input_tensors);
    }
    auto weights = variables_to_idx_vector(node.getWeights());

    /// weights are part of input in tflite
    inputs.insert(inputs.end(), weights.begin(), weights.end());

    auto outputs = variables_to_idx_vector(node.getOutputs());

    auto fb_inputs = fbb.CreateVector(inputs);
    auto fb_outputs = fbb.CreateVector(outputs);
    auto fb_options = node.getBuiltinOps();

    tflite::OperatorBuilder builder(fbb);
    builder.add_opcode_index(op_code);
    builder.add_builtin_options_type(node.getOptionType());
    builder.add_builtin_options(fb_options);
    builder.add_inputs(fb_inputs);
    builder.add_outputs(fb_outputs);
    return builder.Finish();
  };

  std::vector<flatbuffers::Offset<tflite::Operator>> v;
  v.reserve(nodes.size());

  for (auto &node : nodes) {
    if (node->isVirtualNode())
      continue;
    auto op = create_operator(*node);
    v.push_back(op);
  }

  return fbb.CreateVector(v);
}

flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<tflite::SubGraph>>>
buildSubGraphs(const TfOpNodes &nodes, const TfOpIdxMap &map,
               flatbuffers::FlatBufferBuilder &fbb) {

  auto tensors = buildTensors(map, fbb);
  auto ops = buildOperators(nodes, map, fbb);

  /// @todo extract this to buildSubgraph if there is one or more subgraph
  auto name = fbb.CreateString("main");
  auto inputs = fbb.CreateVector(map.getInputs());
  auto outputs = fbb.CreateVector(map.getOutputs());

  auto builder = tflite::SubGraphBuilder(fbb);
  builder.add_tensors(tensors);
  builder.add_inputs(inputs);
  builder.add_outputs(outputs);
  builder.add_name(name);
  builder.add_operators(ops);
  auto subgraph = builder.Finish();

  std::vector<flatbuffers::Offset<tflite::SubGraph>> subgraphs;
  subgraphs.reserve(1);
  subgraphs.push_back(subgraph);

  return fbb.CreateVector(subgraphs);
}

} // namespace

void TfliteInterpreter::serialize(const GraphRepresentation &representation,
                                  const std::string &out) {
  /// @todo check if graph is finalized & initialized and ready to serialize.

  /// 0. remove batch normalization layer in GraphRepresentation
  BnRealizer realizer({});
  GraphRepresentation graph = realizer.realize(representation);

  /// 1. remove loss layer in GraphRepresentation
  LossRealizer loss_realizer({});
  graph = loss_realizer.realize(graph);

  /// 2. The graph must have weights, input dims, output dims set
  flatbuffers::FlatBufferBuilder fbb;

  auto opNodes = buildOpNodes(graph, fbb);
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

GraphRepresentation TfliteInterpreter::deserialize(const std::string &in) {
  /// ======== list of things to consider ========
  /// we need to reconstruct some properties from the shape
  /// eg) units are not saved as a property

  /** NYI! */
  return {};
}

} // namespace nntrainer
