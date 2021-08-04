// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    network_graph.h
 * @date    19 Oct 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network Graph Class for Neural Network
 *
 * @todo    Support multi-input graph.
 */

#include <sstream>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <cross_entropy_loss_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <multiout_layer.h>
#include <network_graph.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <profiler.h>
#include <rnn.h>
#include <split_layer.h>
#include <time_dist.h>

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

int NetworkGraph::compile(const std::string &loss_type) {
  int status = ML_ERROR_NONE;

  status = isCompilable();
  NN_RETURN_STATUS();

  status = realizeGraph();
  NN_RETURN_STATUS();

  graph.realizeInputOutputNode();

  try {
    status = addLossLayer(loss_type);
    NN_RETURN_STATUS();
  } catch (const std::exception &e) {
    ml_loge("%s", e.what());
    status = ML_ERROR_INVALID_PARAMETER;
    NN_RETURN_STATUS();
  }

  graph.topologicalSort();

  countNonTrainableLayersAtBegin();

  status = checkCompiledGraph();
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

void NetworkGraph::updateConnectionName(const std::string &from,
                                        const std::string &to) {
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto &lnode = *iter;
    if (istrequal(lnode->getName(), to))
      continue;
    lnode->updateInputLayers(from, to);
  }
}

void NetworkGraph::addDefaultInputLayers() {
  for (auto iter = cbegin() + 1; iter != cend(); iter++) {
    auto layer = *iter;
    auto prev_layer = *(iter - 1);
    if (layer->getNumInputConnections() == 0) {
      layer->addInputLayers(prev_layer->getName());
    }
  }
}

void NetworkGraph::addLayerNode(std::unique_ptr<Layer> layer) {
  graph.addNode(std::make_unique<LayerNode>(std::move(layer)));
}

void NetworkGraph::countNonTrainableLayersAtBegin() {
  for (auto iter = cbegin(); iter != cend(); iter++) {
    // TODO: check if getTrainable() was set and if trainable weights exist,
    // then throw
    if ((*iter)->getTrainable() && (*iter)->supportBackwarding()) {
      skip_non_trainable_layers = iter - cbegin();
      return;
    }
  }

  skip_non_trainable_layers = graph.size();
}

int NetworkGraph::realizeMultiInputType(
  const std::shared_ptr<LayerNode> &in_node) {
  int status = ML_ERROR_NONE;
  /**
   * Multi-input works with time distribution layer by itself
   *
   */
  if (in_node->getNumInputConnections() <= 1)
    return ML_ERROR_NONE;

  // TODO: this can be addition or concat layer - add support
  std::shared_ptr<LayerNode> lnode = createLayerNode(AdditionLayer::type);
  graph.ensureName(*lnode, in_node->getName());

  lnode->setInputLayers(in_node->getInputLayers());
  in_node->setInputLayers({lnode->getName()});
  /** output layers for layer obj will be set in setOutputLayers() */

  graph.addNode(lnode, false);

  return status;
}

int NetworkGraph::realizeFlattenType(
  const std::shared_ptr<LayerNode> &in_node) {
  if (in_node->getType() == FlattenLayer::type) {
    ml_loge(
      "It is not allowed to realize flatten layer, possibly flatten layer is "
      "added right after flatten");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<LayerNode> lnode = createLayerNode(FlattenLayer::type);
  graph.ensureName(*lnode, in_node->getName());

  lnode->setInputLayers({in_node->getName()});
  /** output layers for layer obj will be set in setOutputLayers() */

  updateConnectionName(in_node->getName(), lnode->getName());
  graph.addNode(lnode, false);

  return ML_ERROR_NONE;
}

int NetworkGraph::realizeActivationType(
  const std::shared_ptr<LayerNode> &in_node) {
  int status = ML_ERROR_NONE;

  ActivationType act = in_node->getActivationToBeRealized();

  if (act == ActivationType::ACT_NONE) {
    /// ActivationType::ACT_NONE does not need realization
    return ML_ERROR_NONE;
  }

  if (act == ActivationType::ACT_UNKNOWN) {
    ml_loge("cannot realize unknown activation type");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (in_node->getType() == ActivationLayer::type) {
    ml_loge("It is not allowed to realize activation layer, possibly layer is "
            "added right after activation");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<LayerNode> lnode = createLayerNode(ActivationLayer::type);
  graph.ensureName(*lnode, in_node->getName());

  if (in_node->getDistribute()) {
    lnode->setProperty({"distribute=true"});
  }

  lnode->setProperty({"activation=" + ActivationTypeStr[(unsigned int)act]});
  in_node->setProperty({"activation=none"});

  lnode->setInputLayers({in_node->getName()});
  /** output layers for layer obj will be set in setOutputLayers() */

  updateConnectionName(in_node->getName(), lnode->getName());
  graph.addNode(lnode, false);

  return status;
}

int NetworkGraph::realizeMultiOutputType(
  const std::shared_ptr<LayerNode> &in_node) {
  int status = ML_ERROR_NONE;
  /**
   * Multi-input works with time distribution layer by itself
   *
   */

  if (in_node->getNumOutputConnections() <= 1)
    return ML_ERROR_NONE;

  std::shared_ptr<LayerNode> lnode = createLayerNode(MultiOutLayer::type);
  graph.ensureName(*lnode, in_node->getName());

  lnode->setInputLayers({in_node->getName()});
  lnode->setOutputLayers(in_node->getOutputLayers());

  in_node->setOutputLayers({lnode->getName()});

  for (unsigned int i = 0; i < in_node->getNumOutputConnections(); ++i) {
    updateConnectionName(in_node->getName(), lnode->getName());
  }

  graph.addNode(lnode, false);

  return status;
}

int NetworkGraph::addLossLayer(const std::string &loss_type_) {
  for (unsigned int i = 0; i < graph.getNumOutputNodes(); ++i) {
    auto output_layer_node = LNODE(graph.getOutputNode(i));
    std::string loss_type = loss_type_;

    if (output_layer_node->requireLabel())
      continue;

    if (loss_type.empty())
      continue;

    auto second_to_last_layer_node = output_layer_node;
    bool is_cross_entropy_loss =
      istrequal(loss_type, CrossEntropyLossLayer::type);
    if (is_cross_entropy_loss) {
      auto type = output_layer_node->getType();

      if (type != ActivationLayer::type) {
        throw exception::not_supported(
          "Error: Cross Entropy need last layer to have softmax or sigmoid"
          "activation.");
      }

      switch (output_layer_node->getActivationType()) {
      case ActivationType::ACT_SIGMOID:
        loss_type = CrossEntropySigmoidLossLayer::type;
        break;
      case ActivationType::ACT_SOFTMAX:
        loss_type = CrossEntropySoftmaxLossLayer::type;
        break;
      default:
        throw exception::not_supported(
          "Error: Cross Entropy not supported without softmax or sigmoid.");
      }

      second_to_last_layer_node =
        LNODE(graph.getNode(output_layer_node->getInputLayers()[0]));
    }

    std::shared_ptr<LayerNode> lnode = createLayerNode(loss_type);
    graph.ensureName(*lnode);

    if (second_to_last_layer_node->getDistribute()) {
      lnode->setProperty({"distribute=true"});
    }

    second_to_last_layer_node->setOutputLayers({lnode->getName()});
    lnode->setInputLayers({second_to_last_layer_node->getName()});

    if (is_cross_entropy_loss) {
      graph.replaceNode(output_layer_node, lnode);
    } else {
      graph.addNode(lnode, false);
    }
    graph.replaceOutputNode(i, lnode);
  }

  return ML_ERROR_NONE;
}

void NetworkGraph::setOutputLayers() {

  for (auto iter_idx = cbegin(); iter_idx != cend(); iter_idx++) {
    auto &layer_idx = *iter_idx;
    for (auto iter_i = cbegin(); iter_i != cend(); iter_i++) {
      auto &layer_i = *iter_i;
      if (istrequal(layer_i->getName(), layer_idx->getName()))
        continue;
      for (unsigned int j = 0; j < layer_i->getNumInputConnections(); ++j) {
        if (istrequal(layer_i->getInputLayers()[j], layer_idx->getName())) {
          bool already_exist = false;
          for (unsigned int k = 0; k < layer_idx->getNumOutputConnections();
               ++k) {
            if (istrequal(layer_idx->getOutputLayers()[k],
                          layer_i->getName())) {
              already_exist = true;
              break;
            }
          }

          if (!already_exist)
            layer_idx->addOutputLayers(layer_i->getName());
        }
      }
    }
  }
}

int NetworkGraph::isCompilable() {
  if (compiled) {
    ml_loge("Graph is already compiled");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (graph.empty()) {
    ml_loge("Graph is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

int NetworkGraph::checkCompiledGraph() {
  /** Dimension of input layers must be known */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto lnode = (*iter);
    if (lnode->getType() == InputLayer::type) {
      if (lnode->getInputDimensions().size() == 0) {
        ml_loge("InputDimension of first layer is not set");
        return ML_ERROR_INVALID_PARAMETER;
      }
    }
  }

  return ML_ERROR_NONE;
}

int NetworkGraph::realizeGraph() {

  int status = ML_ERROR_NONE;

  addDefaultInputLayers();

  /**
   * invariant: the new realized nodes are added to the end,
   * otherwise this iteration becomes invalid. So, every iteration must be fresh
   * iterator as vector resize invalidates all the iterators.
   */
  for (unsigned int i = 0; i < graph.size(); ++i) {
    auto const &lnode = LNODE(*(cbegin() + i));
    ml_logd("layer name: %s", lnode->getName().c_str());

    /** If a layer does not has input nodes, then it must have input dimension
     */
    if (lnode->getNumInputConnections() == 0) {
      for (unsigned int i = 0; i < lnode->getInputDimensions().size(); ++i) {
        if (lnode->getInputDimensions()[i].getDataLen() == 0) {
          ml_loge("Input Dimension must be set");
          status = ML_ERROR_INVALID_PARAMETER;
          NN_RETURN_STATUS();
        }
      }
    }

    if (lnode->getType() != AdditionLayer::type &&
        lnode->getType() != ConcatLayer::type) {
      status = realizeMultiInputType(lnode);
      NN_RETURN_STATUS();
    }

    if (lnode->getType() != ActivationLayer::type) {
      status = realizeActivationType(lnode);
      NN_RETURN_STATUS();
    }

    // Flatten in TimeDistLayer is not supported.
    if (lnode->getFlatten() && !lnode->getDistribute()) {
      status = realizeFlattenType(lnode);
      NN_RETURN_STATUS();
    }
  }

  try {
    setOutputLayers();
  } catch (std::exception &e) {
    ml_loge("setting output layer failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  /**
   * invariant: the new realized nodes are added to the end,
   * otherwise this iteration becomes invalid. So, every iteration must be fresh
   * iterator as vector resize invalidates all the iterators.
   */
  for (unsigned int i = 0; i < graph.size(); ++i) {
    auto const &lnode = LNODE(*(cbegin() + i));
    if (lnode->getType() != MultiOutLayer::type &&
        lnode->getType() != SplitLayer::type) {
      status = realizeMultiOutputType(lnode);
      NN_RETURN_STATUS();
    }
  }
  /// @todo add check that input_layers <-> output_layers does match.
  /// @todo check whether graph has a cycle or graph is seperated to subgraph

  return status;
}

void NetworkGraph::setBatchSize(unsigned int batch_size) {
  for (auto iter = cbegin(); iter != cend(); iter++) {
    (*iter)->setBatch(batch_size);
  }
}

sharedConstTensors NetworkGraph::forwarding(bool training) const {
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto const &ln = *iter;
    START_PROFILE(ln->event_key);
    ln->forwarding(training);
    END_PROFILE(ln->event_key);
  }

  sharedConstTensors out;
  for (unsigned int i = 0; i < graph.getNumOutputNodes(); ++i) {
    auto const &output_layer_node = LNODE(graph.getOutputNode(i));
    for (unsigned int j = 0; j < output_layer_node->getNumOutputs(); ++j) {
      out.push_back(MAKE_SHARED_TENSOR(output_layer_node->getOutput(j)));
    }
  }

  return out;
}

std::vector<TensorDim> NetworkGraph::getInputDimension() const {
  NNTR_THROW_IF(this->empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node!";
  return getSortedLayerNode(0)->getInputDimensions();
}

std::vector<TensorDim> NetworkGraph::getOutputDimension() const {
  NNTR_THROW_IF(this->empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node!";
  return getSortedLayerNode(graph.size() - 1)->getOutputDimensions();
}

std::vector<std::shared_ptr<LayerNode>>
NetworkGraph::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {
  /// @fixme: this won't work if input, output layers are not in order
  /// Further, this function must be removed. There should be rather
  /// getAllNames and getLayerByName instead of getUnsortedLayers.

  /** count layers after output layer */
  unsigned int num_layers_remove_end = 0;
  if (!output_layer.empty()) {
    for (auto iter = graph.crbegin(); iter != graph.crend(); iter++) {
      if ((*iter)->getName() != output_layer)
        num_layers_remove_end++;
      else
        break;
    }
  }

  if (num_layers_remove_end == graph.size())
    return {};

  /** count layers before input layer */
  unsigned int num_layers_remove_start = 0;
  if (!input_layer.empty()) {
    for (auto iter = graph.cbegin();
         iter != graph.cend() - num_layers_remove_end; iter++) {
      if ((*iter)->getName() != input_layer)
        num_layers_remove_start++;
      else
        break;
    }
  }

  /** copy the graph and return */
  std::vector<std::shared_ptr<LayerNode>> ret;
  std::transform(graph.cbegin() + num_layers_remove_start,
                 graph.cend() - num_layers_remove_end, std::back_inserter(ret),
                 [](auto const &elem) { return LNODE(elem); });

  return ret;
}

std::vector<std::shared_ptr<LayerNode>> NetworkGraph::getLayerNodes() const {
  return std::vector<std::shared_ptr<LayerNode>>(cbegin(), cend());
}

void NetworkGraph::extendGraph(std::vector<std::shared_ptr<LayerNode>> ex_graph,
                               std::string &prefix) {

  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /**
   * The input_layers for ex_graph[0] here is provided to the backbone by the
   * ini file and is overwritten here by the model loader for connection making.
   *
   * This loop intends to connect a new backbone to be added with an old
   * backbone.
   */
  auto &layer0_in = ex_graph[0]->getInputLayers();
  for (unsigned int i = 0; i < layer0_in.size(); ++i) {
    if (sub_in_out.find(layer0_in[i]) != sub_in_out.end()) {
      ex_graph[0]->updateInputLayers(i, sub_in_out[layer0_in[i]]);
    } else if (!graph.verifyNode(layer0_in[i])) {
      throw std::runtime_error("Input layer name for backbone not found.");
    }
  }

  /** Insert the layer to the graph */
  for (auto &layernode : ex_graph) {
    /**
     * Add prefix to the existing layer name,
     * and ensure it is unique in this new ex_graph
     */
    std::string orig_name = prefix + layernode->getName();
    graph.ensureName(*layernode, prefix, "", true);
    sub_in_out.insert(std::make_pair(orig_name, layernode->getName()));

    auto &input_layers = layernode->getInputLayers();
    for (unsigned int i = 0; i < input_layers.size(); ++i) {
      if (sub_in_out.find(prefix + input_layers[i]) != sub_in_out.end()) {
        layernode->updateInputLayers(
          i, sub_in_out[prefix + layernode->getInputLayers()[i]]);
      } else if (!graph.verifyNode(layernode->getInputLayers()[i])) {
        throw std::runtime_error("Input layer name for backbone not found.");
      }
    }

    graph.addNode(layernode, false);
  }

  /** This allows connecting a layer to the backbone */
  sub_in_out.insert(
    std::make_pair(prefix, graph.getNode(graph.size() - 1)->getName()));
}

void NetworkGraph::addLayer(std::shared_ptr<LayerNode> layer) {
  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /** Insert the layer to the graph */
  graph.addNode(layer);
}

void NetworkGraph::inPlaceOptimize(Manager &manager) {
  // TODO: update this after initial verification, this is deprecated for now.
  return;

#if 0
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto layer_node = *iter;
    auto &l = layer_node->getObject();
    std::string l_type = l->getType();

    /**
     * @todo concat/multi-output layer can be made in-place but with special
     * handling where these layers can become no-op.
     * consider this optimization for later
     */

    if (l->supportInPlace()) {
      /** @note assumes layer to be optimized is only for single in/out tensor
       */
      if (layer_node->getNumInputConnections() != 1)
        throw std::runtime_error("Internal error in the formed graph");

      auto prev_node = getLayerNode(layer_node->getInputLayers()[0]);
      auto &prev_layer =
        getLayerNode(layer_node->getInputLayers()[0])->getObject();

      unsigned int loc;
      auto layer_name = layer_node->getName();
      auto &output_layers = prev_node->getOutputLayers();
      for (loc = 0; loc < output_layers.size(); ++loc)
        if (output_layers[loc] == layer_name)
          break;

      if (loc == output_layers.size())
        throw std::runtime_error("Internal error in the formed graph.");

      /** Previous layer cannot be input layer for in-place layer */
      if (prev_node->getType() == InputLayer::type)
        continue;

      /** Two layers cant work in-place consecutively */
      if (prev_node->supportInPlace())
        continue;

      /** Share tensor with next layer */
      /**
       * Assume two layers, L1 and L2, with I and O corresponding to the layer
       * outputs. Assume L2 to be the layer, needing in-place optimization.
       */
      if (l_type == BatchNormalizationLayer::type) {
        /**
         * With batch normalization, neither input nor output of the layer are
         * requried for calculatin gradient and derivative. Just input
         * derivative is required. In this scenraio, L2 is assumed to be batch
         * normalization layer, and L1 is assumed to be a non-in-place layer.
         * Hence, L1 layer's output and L2 layer's input var_grad are modified.
         */
        auto &inplace_shared_vg_ptr = l->net_hidden[0];
        l->net_input[0] = inplace_shared_vg_ptr;             /// I2 = O2
        prev_layer->net_hidden[loc] = inplace_shared_vg_ptr; /// O1 = O2
      } else if (l_type == ActivationLayer::type) {
        /**
         * For activation layer, output of the layer and input derivative, both
         * , are requried for calculating the gradient and derivative. In this
         * scenraio, L2 is assumed to be activation layer, and L1 is assumed to
         * be a non-in-place layer.
         * Hence, L1 layer's output and L2 layer's input var_grad are
         * differently. L1 layer operates out of place and share the memory for
         * its output (O1.V) and input derivative (O1.G). L2 layer operates
         * in-place and use a common shared derivative memory for their
         * derivatives (handled in manager.cpp).
         */
        /**
         * @note As this updates the tensors used in the prev_layer->net_hidden
         * (O1), the tensors inside l->net_input (I2) are automatically updated
         * as they share the same var_grad object.
         */
        auto &inplace_shared_vg = *l->net_hidden[0].get();
        prev_layer->net_hidden[loc]->updateVariableByVariable(
          inplace_shared_vg); /// O1.G = O2.V
        prev_layer->net_hidden[loc]->updateGradientByVariable(
          inplace_shared_vg); /// O1.V = O2.V
      } else {
        std::stringstream ss;
        ss << l_type;
        ss << " layer is not supported for in-place optimization";
        throw std::runtime_error(ss.str());
      }

      /** Untrack the memory for this layer */
      manager.untrackLayerInOuts(prev_node->getName());
    }
  }
#endif
}

std::vector<Var_Grad *>
NetworkGraph::updateRunContext(std::shared_ptr<Manager> &manager,
                               const std::shared_ptr<LayerNode> &lnode,
                               const std::vector<Var_Grad *> &prev_inputs) {
  /**
   * using copy assignment allows setting run_context without adding more
   * interfaces
   */
  const GraphNode &gnode = *lnode.get();
  const InitLayerContext &init_context = lnode->getInitContext();
  std::vector<Var_Grad *> inputs = prev_inputs;
  if (inputs.empty())
    inputs = manager->requestInputs(gnode, init_context.getInputDimensions());

  const std::vector<Var_Grad *> &outputs =
    manager->requestOutputs(gnode, init_context.getOutputDimensions());

  /**
   * @note must use existing properties like name/trainable of run_context to
   * create the new run_context
   */
  const RunLayerContext &run_context = lnode->getRunContext();
  lnode->updateRunContext(RunLayerContext(
    run_context.getName(), run_context.getLoss(),
    // TODO: update weights spec for trainable based on layer trainable prop
    manager->requestWeights(gnode, init_context.getWeightsSpec()), inputs,
    outputs, manager->requestTensors(gnode, init_context.getTensorsSpec())));

  return outputs;
}

int NetworkGraph::initialize(std::shared_ptr<Manager> manager) {
  int status = ML_ERROR_NONE;
  /** this contains the map from name to input tensors for each node */
  std::unordered_map<std::string, std::vector<Var_Grad *>> input_map;

  /** check if the given config of node is of input node */
  auto is_input_node = [](const std::string &type,
                          const unsigned int idx) -> bool {
    /** TODO: remove dependency on idx */
    return type == InputLayer::type || idx == 0;
  };

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    auto const &lnode = getSortedLayerNode(idx);
    std::string cur_type = lnode->getType();
    ml_logd("layer name : %s", lnode->getName().c_str());

    /**
     * Set input dimension for all the layers.
     * For input layer, as input dimension is known, set input tensor.
     */
    if (!is_input_node(cur_type, idx)) {
      auto &input_layers = lnode->getInputLayers();
      for (unsigned int i = 0; i < input_layers.size(); ++i) {
        auto in_layer_node = getLayerNode(input_layers[i]);

        auto const &in_layer_out_connect = in_layer_node->getOutputLayers();
        unsigned int location =
          std::find(in_layer_out_connect.begin(), in_layer_out_connect.end(),
                    lnode->getName()) -
          in_layer_out_connect.begin();

        lnode->setInputDimension(in_layer_node->getOutputDimensions()[location],
                                 i);
      }
    }

    /**
     * Initialize all the layers, allocate output tensors for each layer
     * init2and add optimizer related weights for the layer
     */
    lnode->finalize();

    std::vector<Var_Grad *> inputs = {};
    if (!is_input_node(cur_type, idx)) {
      if (input_map.find(lnode->getName()) == input_map.end())
        throw std::runtime_error("Cannot find input buffers for the node");
      inputs = input_map.at(lnode->getName());
    }
    const std::vector<Var_Grad *> &outputs =
      updateRunContext(manager, lnode, inputs);

    /** no need to update input_map for the last layer */
    if (idx == graph.size() - 1)
      break;

    auto &output_layers = lnode->getOutputLayers();
    for (unsigned int i = 0; i < output_layers.size(); ++i) {
      auto out_layer_node = getLayerNode(output_layers[i]);
      if (input_map.find(output_layers[i]) == input_map.end())
        input_map.insert({output_layers[i], {}});

      unsigned int j = 0;
      for (; j < out_layer_node->getNumInputConnections(); ++j) {
        if (istrequal(out_layer_node->getInputLayers()[j], lnode->getName())) {
          break;
        }
      }

      auto &in_map = input_map.at(output_layers[i]);
      in_map.resize(out_layer_node->getNumInputConnections());
      in_map[j] = outputs[i];
    }
  }
  return status;
}

} /* namespace nntrainer */
