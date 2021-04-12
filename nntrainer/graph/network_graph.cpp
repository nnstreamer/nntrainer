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
#include <flatten_layer.h>
#include <input_layer.h>
#include <layer_factory.h>
#include <loss_layer.h>
#include <network_graph.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <output_layer.h>
#include <parse_util.h>
#include <profiler.h>
#include <rnn.h>
#include <time_dist.h>

namespace nntrainer {

/**
 * @todo Make inPlace as a static property of the layer and a state to verify if
 * this layer is working in-place
 */
static const std::vector<std::string> in_place_layers = {
  ActivationLayer::type, BatchNormalizationLayer::type};

static std::shared_ptr<Layer> distributeLayer(std::shared_ptr<Layer> l) {
  std::shared_ptr<Layer> layer = nntrainer::createLayer(TimeDistLayer::type);
  std::dynamic_pointer_cast<TimeDistLayer>(layer)->setDistLayer(l);

  return layer;
}

int NetworkGraph::compile(const LossType loss_type) {
  int status = ML_ERROR_NONE;

  status = isCompilable();
  NN_RETURN_STATUS();

  status = realizeGraph();
  NN_RETURN_STATUS();

  status = connectGraph();
  NN_RETURN_STATUS();

  topologicalSort();

  status = addLossLayer(loss_type);
  NN_RETURN_STATUS();

  status = checkCompiledGraph();
  NN_RETURN_STATUS();

  /**
   * Now that graph is compiled, remove all edges to save memory.
   * NodeList is kept for now for O(1) access of layers by idx.
   */
  for (unsigned int i = 0; i < adj.size(); ++i)
    /**
     * As this resize is guaranteed to not insert new elements,  create a
     * default element needed by resize.
     */
    adj[i].resize(1, LayerNode(nullptr, 0));

  compiled = true;

  return status;
}

void NetworkGraph::updateConnectionName(const std::string &from,
                                        const std::string &to) {
  for (unsigned int i = 0; i < adj.size(); ++i) {
    auto &layer = adj[i].front().getObject();
    if (istrequal(layer->getName(), to))
      continue;
    for (unsigned int j = 0; j < layer->input_layers.size(); ++j) {
      if (istrequal(layer->input_layers[j], from)) {
        layer->input_layers[j] = to;
      }
    }
  }
}

void NetworkGraph::addDefaultInputLayers() {
  for (unsigned int i = 1; i < adj.size(); ++i) {
    auto &layer = adj[i].front().getObject();
    auto &prev_layer = adj[i - 1].front().getObject();
    if (layer->input_layers.size() == 0) {
      layer->input_layers.push_back(prev_layer->getName());
    }
  }
}

void NetworkGraph::addLayerNode(std::shared_ptr<Layer> layer) {
  std::list<LayerNode> l;
  std::unique_ptr<LayerNode> node =
    std::make_unique<LayerNode>(layer, adj.size());

  l.assign(1, *node);
  adj.push_back(l);
}

LayerNode &NetworkGraph::getLayerNode(unsigned int ith) {
  if (ith >= size())
    throw std::invalid_argument("Exceed total number of layer");

  if (adj[ith].front().getIndex() != ith)
    throw std::runtime_error("Graph internal index mismatch");

  return adj[ith].front();
}

LayerNode &NetworkGraph::getSortedLayerNode(unsigned int ith) {
  if (ith >= getSorted().size())
    throw std::invalid_argument("Exceed total number of layer");

  return getSorted()[ith];
}

void NetworkGraph::topologicalSortUtil(unsigned int ith,
                                       std::vector<bool> &visited,
                                       std::stack<LayerNode> &Stack) {
  visited[ith] = true;

  std::list<LayerNode>::iterator i;
  for (i = adj[ith].begin(); i != adj[ith].end(); ++i) {
    auto index = (*i).getIndex();
    if (!visited[index])
      topologicalSortUtil(index, visited, Stack);
  }

  Stack.push(getLayerNode(ith));
}

void NetworkGraph::countNonTrainableLayersAtBegin() {
  for (auto iter = Sorted.cbegin(); iter != Sorted.cend(); iter++) {
    if ((*iter).getObject()->getTrainable()) {
      skip_non_trainable_layers = iter - Sorted.cbegin();
      break;
    }
  }
}

void NetworkGraph::topologicalSort() {
  std::stack<LayerNode> Stack;
  std::vector<bool> visited(adj.size());
  Sorted.clear();

  std::fill(visited.begin(), visited.end(), false);

  // TODO : After make node list of graph, we have to find root. (That means it
  // should be the only one input for now.). Need to support multiple input and
  // support search.

  for (unsigned int i = 0; i < adj.size(); ++i) {
    if (visited[i] == false) {
      topologicalSortUtil(i, visited, Stack);
    }
  }

  while (Stack.empty() == false) {
    Sorted.push_back(Stack.top());
    Stack.pop();
  }

  countNonTrainableLayersAtBegin();
}

void NetworkGraph::ensureName(std::shared_ptr<Layer> layer,
                              const std::string &prefix,
                              const std::string &postfix, bool force_rename) {
  std::string orig_name = layer->getName();
  bool orig_name_empty = orig_name.empty();
  /** If layer already has name which is unique and valid, and force is
   * disabled, then nothing to do.
   */
  if (!orig_name_empty && !force_rename &&
      layer_names.end() == layer_names.find(orig_name)) {
    layer_names.insert(orig_name);
    return;
  }

  /** If just prefix with layer name makes it unique - directly set the name */
  if (!orig_name_empty) {
    std::string direct_name = prefix + orig_name + postfix;
    if (layer_names.find(direct_name) == layer_names.end()) {
      layer->setName(direct_name);
      layer_names.insert(direct_name);
      return;
    }
  }

  std::set<std::string>::iterator iter;
  std::string name;
  if (orig_name_empty) {
    orig_name = layer->getType();
  }

  std::string direct_name = prefix + orig_name + postfix;

  do {
    name = direct_name + std::to_string(def_name_count++);
    iter = layer_names.find(name);
  } while (iter != layer_names.end());

  layer->setName(name);
  layer_names.insert(name);
}

int NetworkGraph::realizeMultiInputType(Layer &current) {
  int status = ML_ERROR_NONE;
  if (current.getNumInputs() == 1)
    return ML_ERROR_NONE;

  // TODO: this can be addition or concat layer - add support
  std::shared_ptr<Layer> layer = nntrainer::createLayer(AdditionLayer::type);
  ensureName(layer, current.getName());

  layer->setNumInputs(current.getNumInputs());
  layer->input_layers.clear();
  for (unsigned int i = 0; i < current.input_layers.size(); ++i)
    layer->input_layers.push_back(current.input_layers[i]);

  layer->setNumOutputs(current.getNumOutputs());

  current.setNumInputs(layer->getNumOutputs());
  current.input_layers.clear();
  current.input_layers.push_back(layer->getName());
  /** output layers for layer obj will be set in setOutputLayers() */

  addLayerNode(layer);

  return status;
}

int NetworkGraph::realizeFlattenType(Layer &current) {
  if (adj.empty()) {
    ml_loge("layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (current.getType() == FlattenLayer::type) {
    ml_loge(
      "It is not allowed to realize flatten layer, possibly flatten layer is "
      "added right after flatten");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<Layer> layer = nntrainer::createLayer(FlattenLayer::type);

  ensureName(layer, current.getName());

  layer->setNumInputs(current.getNumInputs());
  layer->input_layers.clear();
  layer->input_layers.push_back(current.getName());
  layer->setNumOutputs(current.getNumOutputs());
  /** output layers for layer obj will be set in setOutputLayers() */

  updateConnectionName(current.getName(), layer->getName());
  addLayerNode(layer);

  return ML_ERROR_NONE;
}

int NetworkGraph::realizeActivationType(Layer &current) {
  int status = ML_ERROR_NONE;

  ActivationType act = current.getActivationType();

  if (current.getType() == RNNLayer::type) {
    // No need to add activation layer for RNN Layer
    // Default activation is tanh
    if (act == ActivationType::ACT_NONE)
      act = ActivationType::ACT_TANH;
    current.setActivation(act);
    return status;
  }

  if (act == ActivationType::ACT_NONE) {
    /// ActivationType::ACT_NONE does not need realization
    return ML_ERROR_NONE;
  }

  if (adj.empty()) {
    ml_loge("layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (current.getType() == ActivationLayer::type) {
    ml_loge("It is not allowed to realize ativation layer, possibly layer is "
            "added right after activation");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (act == ActivationType::ACT_UNKNOWN) {
    ml_loge("cannot realize unknown activation type");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<Layer> layer = nntrainer::createLayer(ActivationLayer::type);
  layer->setActivation(act);

  ensureName(layer, current.getName());

  if (current.getType() == TimeDistLayer::type) {
    std::string unit_str = layer->getName();
    ensureName(layer, "", "_unit");
    layer = distributeLayer(layer);
    layer->setName(unit_str);
  }

  layer->setNumInputs(current.getNumInputs());
  layer->input_layers.clear();
  layer->input_layers.push_back(current.getName());
  layer->setNumOutputs(current.getNumOutputs());
  /** output layers for layer obj will be set in setOutputLayers() */

  updateConnectionName(current.getName(), layer->getName());
  addLayerNode(layer);

  return status;
}

int NetworkGraph::realizeMultiOutputType(Layer &current) {
  int status = ML_ERROR_NONE;
  if (current.getNumOutputs() == 1)
    return ML_ERROR_NONE;

  std::shared_ptr<Layer> layer = nntrainer::createLayer(OutputLayer::type);
  ensureName(layer, current.getName());

  layer->setNumInputs(current.getNumInputs());
  layer->input_layers.clear();
  layer->input_layers.push_back(current.getName());
  layer->setNumOutputs(current.output_layers.size());
  /** output layers for layer obj will be set in setOutputLayers() */

  for (unsigned int i = 0; i < current.output_layers.size(); ++i) {
    updateConnectionName(current.getName(), layer->getName());
  }

  current.setNumOutputs(layer->getNumInputs());

  addLayerNode(layer);

  return status;
}

int NetworkGraph::addLossLayer(const LossType loss_type) {

  int status = ML_ERROR_NONE;

  if (Sorted.back().getObject()->getType() == LossLayer::type)
    return status;

  if (Sorted.back().getObject()->getType() == TimeDistLayer::type) {
    if (std::static_pointer_cast<TimeDistLayer>(Sorted.back().getObject())
          ->getDistLayerType() == LossLayer::type)
      return status;
  }

  if (loss_type == LossType::LOSS_NONE) {
    return ML_ERROR_NONE;
  }

  LossType updated_loss_type = loss_type;
  if (adj.empty()) {
    status = ML_ERROR_INVALID_PARAMETER;
    NN_RETURN_STATUS();
  }

  LayerNode last_node = Sorted.back();
  if (updated_loss_type == LossType::LOSS_ENTROPY) {
    auto type = last_node.getObject()->getType();
    if (type == TimeDistLayer::type) {
      type = std::dynamic_pointer_cast<TimeDistLayer>(last_node.getObject())
               ->getDistLayerType();
    }

    if (type != "activation") {
      ml_loge("Error: Cross Entropy need last layer to have softmax or sigmoid"
              "activation.");
      return ML_ERROR_NOT_SUPPORTED;
    }

    LayerNode last_node = Sorted.back();
    adj.pop_back();
    Sorted.pop_back();

    switch (last_node.getObject()->getActivationType()) {
    case ActivationType::ACT_SIGMOID:
      updated_loss_type = LossType::LOSS_ENTROPY_SIGMOID;
      break;
    case ActivationType::ACT_SOFTMAX:
      updated_loss_type = LossType::LOSS_ENTROPY_SOFTMAX;
      break;
    default:
      ml_loge("Error: Cross Entropy not supported without softmax or sigmoid.");
      return ML_ERROR_NOT_SUPPORTED;
    }
  }

  last_node = Sorted.back();
  std::string input_str = last_node.getObject()->getName();

  std::shared_ptr<Layer> layer = nntrainer::createLayer(LossLayer::type);

  status =
    std::dynamic_pointer_cast<LossLayer>(layer)->setLoss(updated_loss_type);
  NN_RETURN_STATUS();

  ensureName(layer);

  if (last_node.getObject()->getType() == TimeDistLayer::type) {
    std::string unit_str = layer->getName();
    ensureName(layer, "", "_unit");
    layer = distributeLayer(layer);
    layer->setName(unit_str);
  }

  last_node.getObject()->setNumOutputs(1);
  last_node.getObject()->output_layers.clear();
  last_node.getObject()->output_layers.push_back(layer->getName());

  layer->setNumInputs(1);
  layer->input_layers.clear();
  layer->input_layers.push_back(input_str);

  /** Set output layers here as setOutputLayers will not be called after adding
   * loss. */
  if (layer->output_layers.size() == 0) {
    layer->setNumOutputs(1);
    layer->output_layers.push_back("__exit__");
  }

  /**
   * As the loss layer is always the last, it could be added manually to Sorted
   * for performance.
   */
  addLayerNode(layer);
  connectGraph(adj.size() - 1);
  Sorted.push_back(adj.back().front());

  return ML_ERROR_NONE;
}

void NetworkGraph::setOutputLayers() {

  size_t last_layer_count = 0;
  for (unsigned int idx = 0; idx < adj.size(); ++idx) {
    auto &layer_idx = adj[idx].front().getObject();
    for (unsigned int i = 0; i < adj.size(); ++i) {
      auto &layer_i = adj[i].front().getObject();
      if (istrequal(layer_i->getName(), layer_idx->getName()))
        continue;
      for (unsigned int j = 0; j < layer_i->input_layers.size(); ++j) {
        if (istrequal(layer_i->input_layers[j], layer_idx->getName())) {
          for (unsigned int k = 0; k < layer_idx->output_layers.size(); ++k) {
            if (!istrequal(layer_idx->output_layers[k], layer_i->getName()))
              continue;
          }
          layer_idx->output_layers.push_back(layer_i->getName());
        }
      }
    }

    if (layer_idx->getNumOutputs() != layer_idx->output_layers.size()) {
      if (layer_idx->output_layers.size() == 0) {
        /** No output layer inplies its the last layer */
        layer_idx->setNumOutputs(1);
        layer_idx->output_layers.clear();
        layer_idx->output_layers.push_back("__exit__");
        last_layer_count += 1;
      } else if (layer_idx->getNumOutputs() < layer_idx->output_layers.size()) {
        /** this if the multi-output layer */
        if (layer_idx->getType() != OutputLayer::type)
          throw std::logic_error("Error: Graph has more edges than expected.");
        layer_idx->setNumOutputs(layer_idx->output_layers.size());
      } else {
        /** error for any other layer */
        throw std::logic_error("Graph node has fewer edges than expected.");
      }
    }
  }

  if (last_layer_count != 1) {
    throw std::invalid_argument(
      "Error: Multiple last layers in the model not supported");
  }

  for (auto iter = adj.begin(); iter < adj.end(); ++iter) {
    if ((*iter).front().getObject()->output_layers.size() == 0)
      throw std::runtime_error("There is un-connected node");
  }
}

int NetworkGraph::isCompilable() {
  if (compiled) {
    ml_loge("Graph is already compiled");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (adj.empty()) {
    ml_loge("Layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

int NetworkGraph::checkCompiledGraph() {
  auto &l = Sorted[0].getObject();
  /** First layer cannot be activation, batch normalization or loss */
  const std::string &type = l->getType();
  if (istrequal(type, ActivationLayer::type) ||
      istrequal(type, BatchNormalizationLayer::type) ||
      istrequal(type, LossLayer::type)) {
    ml_loge("%s cannot be the first layer, type: %s", l->getName().c_str(),
            type.c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** Dimension of input layers must be known */
  for (auto const &lnode : Sorted) {
    if (lnode.getObject()->getType() == InputLayer::type) {
      if (lnode.getObject()->getInputDimension().size() == 0) {
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

  size_t adj_size_before_realize = adj.size();
  /** This loop modifes adj. Get the size of adj preemptively. */

  for (unsigned int i = 0; i < adj_size_before_realize; ++i) {
    Layer &l = *adj[i].front().getObject();
    ml_logd("layer name: %s", l.getName().c_str());

    /** If a layer does not has input nodes, then it must have input dimension
     */
    if (l.input_layers.size() < 1) {
      for (unsigned int i = 0; i < l.getInputDimension().size(); ++i) {
        if (l.getInputDimension()[i].getDataLen() == 0) {
          ml_loge("Input Dimension must be set");
          status = ML_ERROR_INVALID_PARAMETER;
          NN_RETURN_STATUS();
        }
      }

      l.setNumInputs(1);
      l.input_layers.clear();
      l.input_layers.push_back("__data__");
    }

    if (l.getType() != AdditionLayer::type &&
        l.getType() != ConcatLayer::type) {
      status = realizeMultiInputType(l);
      NN_RETURN_STATUS();
    }

    if (l.getType() != ActivationLayer::type) {
      status = realizeActivationType(l);
      NN_RETURN_STATUS();
    }

    if (l.getType() != OutputLayer::type) {
      status = realizeMultiOutputType(l);
      NN_RETURN_STATUS();
    }

    if (l.getFlatten()) {
      status = realizeFlattenType(l);
      NN_RETURN_STATUS();
    }
  }

  setOutputLayers();

  return status;
}

LayerNode &NetworkGraph::getLayerNode(const std::string &layer_name) {

  for (auto &lnode_list : adj) {
    auto &lnode = lnode_list.front();
    if (istrequal(lnode.getObject()->getName(), layer_name))
      return lnode;
  }

  throw std::invalid_argument("Cannot find Layer");
}

void NetworkGraph::addEdge(unsigned int ith, LayerNode &node) {
  if (ith >= adj.size())
    throw std::invalid_argument("Exceed total number of layer");

  adj[ith].push_back(node);
}

void NetworkGraph::connectGraph(unsigned int adj_idx) {
  std::list<LayerNode>::iterator iter = adj[adj_idx].begin();

  for (unsigned int j = 0; j < (*iter).getObject()->input_layers.size(); ++j) {
    if (istrequal((*iter).getObject()->input_layers[j], "__data__"))
      continue;
    unsigned int to_node_id =
      getLayerNode((*iter).getObject()->input_layers[j]).getIndex();
    addEdge(to_node_id, (*iter));
  }
}

int NetworkGraph::connectGraph() {
  for (unsigned int i = 0; i < adj.size(); ++i) {
    connectGraph(i);
  }

  return ML_ERROR_NONE;
}

void NetworkGraph::setBatchSize(unsigned int batch_size) {
  for (auto const &layer_adj_list : adj) {
    layer_adj_list.front().getObject()->setBatch(batch_size);
  }
}

sharedConstTensors NetworkGraph::forwarding(bool training) {
  for (auto const &ln : getSorted()) {
    START_PROFILE(ln.event_key);
    ln.getObject()->forwarding(training);
    END_PROFILE(ln.event_key);
  }

  std::vector<sharedConstTensor> out;
  for (auto const &nh : getSorted().back().getObject()->net_hidden)
    out.push_back(MAKE_SHARED_TENSOR(nh->getVariable()));

  return out;
}

std::vector<TensorDim> NetworkGraph::getInputDimension() const {
  NNTR_THROW_IF(this->empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node!";
  return getSorted()[0].getObject()->getInputDimension();
}

std::vector<TensorDim> NetworkGraph::getOutputDimension() const {
  NNTR_THROW_IF(this->empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node!";
  return getSorted().back().getObject()->getOutputDimension();
}

std::vector<std::shared_ptr<Layer>>
NetworkGraph::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {
  /// @fixme: this won't work if input, output layers are not in order
  /// Further, this function must be removed. There should be rather
  /// getAllNames and getLayerByName instead of getUnsortedLayers.

  /** count layers after output layer */
  unsigned int num_layers_remove_end = 0;
  if (!output_layer.empty()) {
    for (auto iter = adj.rbegin(); iter != adj.rend(); iter++) {
      if ((*iter).front().getObject()->getName() != output_layer)
        num_layers_remove_end++;
      else
        break;
    }
  }

  if (num_layers_remove_end == adj.size())
    return {};

  /** count layers before input layer */
  unsigned int num_layers_remove_start = 0;
  if (!input_layer.empty()) {
    for (auto iter = adj.begin(); iter != adj.end() - num_layers_remove_end;
         iter++) {
      if ((*iter).front().getObject()->getName() != input_layer)
        num_layers_remove_start++;
      else
        break;
    }
  }

  /** copy the graph and return */
  std::vector<std::shared_ptr<Layer>> ret;
  std::transform(adj.begin() + num_layers_remove_start,
                 adj.end() - num_layers_remove_end, std::back_inserter(ret),
                 [](auto const &elem) { return elem.front().getObject(); });

  return ret;
}

std::vector<std::shared_ptr<Layer>> NetworkGraph::getLayers() const {
  std::vector<std::shared_ptr<Layer>> ret;
  if (compiled) {
    std::transform(Sorted.begin(), Sorted.end(), std::back_inserter(ret),
                   [](auto const &elem) { return elem.getObject(); });
  } else {
    std::transform(adj.begin(), adj.end(), std::back_inserter(ret),
                   [](auto const &elem) { return elem.front().getObject(); });
  }

  return ret;
}

void NetworkGraph::extendGraph(std::vector<std::shared_ptr<Layer>> graph,
                               std::string &prefix) {

  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /**
   * The input_layers for graph[0] here is provided to the backbone by the ini
   * file and is overwritten here by the model loader for connection making.
   *
   * This loop intends to connect a new backbone to be added with an old
   * backbone.
   */
  for (unsigned int i = 0; i < graph[0]->input_layers.size(); ++i) {
    if (sub_in_out.find(graph[0]->input_layers[i]) != sub_in_out.end()) {
      graph[0]->input_layers[i] = sub_in_out[graph[0]->input_layers[i]];
    } else if (layer_names.find(graph[0]->input_layers[i]) ==
               layer_names.end()) {
      throw std::runtime_error("Input layer name for backbone not found.");
    }
  }

  /** Insert the layer to the graph */
  for (auto layer : graph) {
    /**
     * Add prefix to the existing layer name,
     * and ensure it is unique in this new graph
     */
    std::string orig_name = prefix + layer->getName();
    ensureName(layer, prefix, "", true);
    sub_in_out.insert(std::make_pair(orig_name, layer->getName()));

    for (unsigned int i = 0; i < layer->input_layers.size(); ++i) {
      if (sub_in_out.find(prefix + layer->input_layers[i]) !=
          sub_in_out.end()) {
        layer->input_layers[i] = sub_in_out[prefix + layer->input_layers[i]];
      } else if (layer_names.find(layer->input_layers[i]) ==
                 layer_names.end()) {
        throw std::runtime_error("Input layer name for backbone not found.");
      }
    }

    addLayerNode(layer);
  }

  /** This allows connecting a layer to the backbone */
  sub_in_out.insert(
    std::make_pair(prefix, adj.back().front().getObject()->getName()));
}

void NetworkGraph::addLayer(std::shared_ptr<Layer> layer) {
  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /** Ensure that the layer has a name and is unique */
  ensureName(layer);

  /** Insert the layer to the graph */
  addLayerNode(layer);
}

void NetworkGraph::inPlaceOptimize(const std::string &layer_type,
                                   Manager &manager) {
  for (auto &layer_node : getSorted()) {
    auto &l = layer_node.getObject();
    if (l->getType() == layer_type &&
        l->getActivationType() != ActivationType::ACT_SOFTMAX) {
      /** @note assumes layer to be optimized is only for single in/out tensor
       */
      if (l->input_layers.size() != 1)
        throw std::runtime_error("Internal error in the formed graph");

      auto &prev_layer = getLayerNode(l->input_layers[0]).getObject();

      unsigned int loc;
      auto layer_name = l->getName();
      for (loc = 0; loc < prev_layer->output_layers.size(); ++loc)
        if (prev_layer->output_layers[loc] == layer_name)
          break;

      if (loc == prev_layer->output_layers.size())
        throw std::runtime_error("Internal error in the formed graph.");

      if (prev_layer->getType() == InputLayer::type)
        continue;

      /** check if previous layer was also in-place */
      bool prev_layer_in_place = false;
      for (auto const &in_place_layer : in_place_layers) {
        if (prev_layer->getType() == in_place_layer) {
          prev_layer_in_place = true;
          break;
        }
      }

      /** Two layers cant work in-place consecutively */
      if (prev_layer_in_place)
        continue;

      /** Share tensor with next layer */
      /**
       * Assume two layers, L1 and L2, with I and O corresponding to the layer
       * outputs. Assume L2 to be the layer, needing in-place optimization.
       */
      if (l->getType() == BatchNormalizationLayer::type) {
        /**
         * With batch normalization, neither input nor output of the layer are
         * requried for calculatin gradient and derivative. Just input
         * derivative is required. In this scenraio, L2 is assumed to be batch
         * normaliztion layer, and L1 is assumed to be a non-in-place layer.
         * Hence, L1 layer's output and L2 layer's input var_grad are modified.
         */
        auto &inplace_shared_vg_ptr = l->net_hidden[0];
        l->net_input[0] = inplace_shared_vg_ptr;             /// I2 = O2
        prev_layer->net_hidden[loc] = inplace_shared_vg_ptr; /// O1 = O2
      } else if (l->getType() == ActivationLayer::type) {
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
        ss << l->getType();
        ss << " layer is not supported for in-place optimization";
        throw std::runtime_error(ss.str());
      }

      /** Untrack the memory for this layer */
      manager.untrackLayerInOuts(prev_layer->getName());
    }
  }
}

void NetworkGraph::inPlaceOptimize(Manager &manager) {
  for (auto const &layer_type : in_place_layers)
    inPlaceOptimize(layer_type, manager);
}

const std::vector<LayerNode> &NetworkGraph::getSorted() const {
  if (!compiled)
    throw std::runtime_error("Cannot get sorted graph before compiling graph");

  return Sorted;
}

std::vector<LayerNode> &NetworkGraph::getSorted() {
  if (!compiled)
    throw std::runtime_error("Cannot get sorted graph before compiling graph");

  return Sorted;
}

} /* namespace nntrainer */
