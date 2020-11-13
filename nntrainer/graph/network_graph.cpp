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
 */

#include <layer_factory.h>
#include <network_graph.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>

namespace nntrainer {

void NetworkGraph::addEdge(unsigned int ith, LayerNode node) {
  if (ith > num_node - 1)
    throw std::invalid_argument("Exceed total number of layer");

  adj[ith].push_back(node);
}

void NetworkGraph::addLayerNode(std::shared_ptr<Layer> layer) {
  std::list<LayerNode> l;
  LayerNode *node = new LayerNode();

  node->layer = layer;
  node->index = num_node;

  l.assign(1, *node);
  adj.push_back(l);
  num_node++;
}

void NetworkGraph::topologicalSortUtil(unsigned int ith, bool visited[],
                                       std::stack<LayerNode> &Stack) {
  visited[ith] = true;

  std::list<LayerNode>::iterator i;
  for (i = adj[ith].begin(); i != adj[ith].end(); ++i) {
    if (!visited[(*i).index])
      topologicalSortUtil((*i).index, visited, Stack);
  }

  Stack.push(getLayerNode(ith));
}

LayerNode &NetworkGraph::getLayerNode(unsigned int ith) {

  std::list<LayerNode>::iterator iter;
  for (unsigned int i = 0; i < num_node; ++i) {
    iter = adj[i].begin();
    if ((*iter).index == ith) {
      return *iter;
    }
  }

  throw std::invalid_argument("Cannot find Layer");
}

LayerNode &NetworkGraph::getSortedLayerNode(unsigned int ith) {

  for (unsigned int i = 0; i < Sorted.size(); ++i) {
    if (Sorted[i].index == ith) {
      return Sorted[i];
    }
  }

  throw std::invalid_argument("Cannot find Layer");
}

void NetworkGraph::topologicalSort() {
  std::stack<LayerNode> Stack;

  bool *visited = new bool[num_node];
  for (unsigned int i = 0; i < num_node; ++i)
    visited[i] = false;

  for (unsigned int i = 0; i < num_node; ++i) {
    if (visited[i] == false) {
      topologicalSortUtil(i, visited, Stack);
    }
  }

  delete[] visited;

  while (Stack.empty() == false) {
    Sorted.push_back(Stack.top());
    Stack.pop();
  }

  for (unsigned int i = 0; i < Sorted.size(); ++i) {
    std::cout << Sorted[i].layer->getName() << " ";
  }
  std::cout << std::endl;
}

void NetworkGraph::ensureName(std::shared_ptr<Layer> layer,
                              const std::string &prefix = "",
                              bool force_rename = false) {
  std::string orig_name = layer->getName();
  bool orig_name_empty = orig_name.empty();
  if (!orig_name_empty && !force_rename &&
      layer_names.end() == layer_names.find(orig_name))
    return;

  /** If just prefix with layer name makes it unique - directly set the name */
  if (!orig_name_empty) {
    std::string direct_name = prefix + orig_name;
    if (layer_names.find(direct_name) == layer_names.end()) {
      layer->setName(direct_name);
      return;
    }
  }

  std::set<std::string>::iterator iter;
  std::string name;
  if (orig_name_empty)
    orig_name = layer->getType();
  std::string direct_name = prefix + orig_name;

  do {
    name = direct_name + std::to_string(def_name_count++);
    iter = layer_names.find(name);
  } while (iter != layer_names.end());

  layer->setName(name);
}

int NetworkGraph::realizeMultiInputType(Layer &current) {
  int status = ML_ERROR_NONE;
  if (current.num_inputs == 1)
    return ML_ERROR_NONE;

  if (current.num_inputs > 1) {
    std::shared_ptr<Layer> layer = nntrainer::createLayer("concat");
    ensureName(layer, current.getName());
    layer->num_inputs = current.num_inputs;
    layer->input_dim.resize(layer->num_inputs);
    layer->input_layers.clear();
    for (unsigned int i = 0; i < current.input_layers.size(); ++i)
      layer->input_layers.push_back(current.input_layers[i]);

    current.num_inputs = 1;
    current.input_layers.clear();
    current.input_layers.push_back(layer->getName());
    addLayerNode(layer);
  }
  return status;
}

int NetworkGraph::realizeFlattenType(
  Layer &current, std::vector<std::shared_ptr<Layer>> layers) {
  if (num_node == 0) {
    ml_loge("layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (current.getType() == "flatten") {
    ml_loge(
      "It is not allowed to realize flatten layer, possibly flatten layer is "
      "added right after flatten");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<Layer> layer = nntrainer::createLayer("flatten");

  ensureName(layer, current.getName());

  layer->num_inputs = 1;
  layer->input_layers.clear();
  layer->input_layers.push_back(current.getName());

  addLayerNode(layer);

  for (unsigned int i = 0; i < layers.size(); ++i) {
    for (unsigned int j = 0; j < layers[i]->input_layers.size(); ++j) {
      if (layers[i]->input_layers[j] == current.getName()) {
        layers[i]->input_layers[j] = layer->getName();
      }
    }
  }

  return ML_ERROR_NONE;
}

int NetworkGraph::realizeActivationType(
  Layer &current, std::vector<std::shared_ptr<Layer>> layers) {
  ActivationType act = current.getActivationType();

  if (act == ActivationType::ACT_NONE) {
    /// ActivationType::ACT_NONE does not need realization
    return ML_ERROR_NONE;
  }

  if (num_node == 0) {
    ml_loge("layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (current.getType() == "activation") {
    ml_loge("It is not allowed to realize ativation layer, possibly layer is "
            "added right after activation");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (act == ActivationType::ACT_UNKNOWN) {
    ml_loge("cannot realize unknown activation type");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<Layer> layer = nntrainer::createLayer("activation");

  ensureName(layer, current.getName());
  layer->setActivation(act);
  layer->num_inputs = 1;
  layer->input_layers.clear();
  layer->input_layers.push_back(current.getName());

  layer->num_outputs = current.num_outputs;
  layer->output_dim.resize(layer->num_outputs);
  layer->output_layers.clear();
  for (unsigned int i = 0; i < current.num_outputs; ++i)
    layer->output_layers.push_back(current.output_layers[i]);

  current.num_outputs = 1;
  current.output_layers.clear();
  current.output_layers.push_back(layer->getName());

  addLayerNode(layer);

  for (unsigned int i = 0; i < layers.size(); ++i) {
    for (unsigned int j = 0; j < layers[i]->input_layers.size(); ++j) {
      if (layers[i]->input_layers[j] == current.getName()) {
        layers[i]->input_layers[j] = layer->getName();
      }
    }
  }

  return ML_ERROR_NONE;
}

int NetworkGraph::addLossLayer(const LossType loss_type) {
  int status = ML_ERROR_NONE;
  LossType updated_loss_type = loss_type;
  if (num_node == 0) {
    status = ML_ERROR_INVALID_PARAMETER;
    NN_RETURN_STATUS();
  }

  if (updated_loss_type == LossType::LOSS_ENTROPY) {
    if (getLayerNode(num_node - 1).layer->getType() != "activation") {
      ml_loge("Error: Cross Entropy need last layer to have softmax or sigmoid "
              "activation.");
      return ML_ERROR_NOT_SUPPORTED;
    }

    LayerNode act_layer_node = getLayerNode(num_node - 1);
    adj.pop_back();
    num_node--;

    switch (act_layer_node.layer->getActivationType()) {
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

  std::string input_str = getLayerNode(num_node - 1).layer->getName();

  std::shared_ptr<Layer> layer = nntrainer::createLayer("loss");

  ensureName(layer);

  LayerNode last_node = getLayerNode(num_node - 1);
  last_node.layer->num_outputs = 1;
  last_node.layer->output_layers.clear();
  last_node.layer->output_layers.push_back(layer->getName());

  layer->num_inputs = 1;
  layer->input_layers.clear();
  layer->input_layers.push_back(input_str);

  if (layer->output_layers.size() == 0) {
    layer->num_outputs = 1;
    layer->output_dim.resize(1);
    layer->output_layers.push_back("exit");
  }

  std::shared_ptr<LossLayer> temp = std::dynamic_pointer_cast<LossLayer>(layer);
  temp->setLoss(updated_loss_type);

  addLayerNode(layer);

  return ML_ERROR_NONE;
}

void NetworkGraph::setOutputLayers(std::vector<std::shared_ptr<Layer>> layers) {

  for (unsigned int idx = 0; idx < layers.size(); ++idx) {
    unsigned int count = 0;
    std::cout << layers[idx]->getName() << " : ";
    for (unsigned int i = 0; i < layers.size(); ++i) {
      if (layers[i]->getName() == layers[idx]->getName())
        continue;
      for (unsigned int j = 0; j < layers[i]->input_layers.size(); ++j) {
        if (layers[i]->input_layers[j] == layers[idx]->getName()) {
          layers[idx]->output_layers.push_back(layers[i]->getName());
          std::cout << layers[idx]->output_layers[count] << ", ";
          count++;
        }
      }
    }
    if (layers[idx]->num_outputs != count) {
      layers[idx]->num_outputs = count;
      layers[idx]->output_dim.resize(count);
    }
    std::cout << std::endl;
  }

  if (layers.back()->num_outputs == 0) {
    layers.back()->num_outputs = 1;
    layers.back()->output_dim.resize(1);
    layers.back()->output_layers.push_back("exit");
  }
}

int NetworkGraph::realizeMultiOutputType(
  Layer &current, std::vector<std::shared_ptr<Layer>> layers) {
  int status = ML_ERROR_NONE;
  if (current.num_outputs == 1)
    return ML_ERROR_NONE;

  if (current.num_outputs > 1) {
    std::shared_ptr<Layer> layer = nntrainer::createLayer("output");
    ensureName(layer, current.getName());

    layer->num_inputs = 1;
    layer->input_layers.clear();
    layer->input_layers.push_back(current.getName());

    layer->num_outputs = current.num_outputs;
    layer->output_layers.clear();

    for (unsigned int i = 0; i < current.output_layers.size(); ++i) {
      layer->output_layers.push_back(current.output_layers[i]);

      for (unsigned int j = 0; j < layers.size(); ++j) {
        for (unsigned int k = 0; k < layers[j]->input_layers.size(); ++k) {
          if (layers[j]->input_layers[k] == current.getName()) {
            layers[j]->input_layers[k] = layer->getName();
          }
        }
      }
    }

    current.num_outputs = 1;
    current.output_layers.clear();
    current.output_layers.push_back(layer->getName());
    addLayerNode(layer);
  }
  return status;
}

int NetworkGraph::setGraphNode(std::vector<std::shared_ptr<Layer>> layers,
                               const LossType loss_type) {

  int status = ML_ERROR_NONE;

  setOutputLayers(layers);

  for (unsigned int i = 0; i < layers.size(); ++i) {
    Layer &l = *layers[i];
    ml_logd("layer name: %s", l.getName().c_str());

    if (l.getType() != "addition" || l.getType() != "concat") {
      status = realizeMultiInputType(l);
      NN_RETURN_STATUS();
    }

    addLayerNode(layers[i]);

    if (l.getType() != "activation") {
      status = realizeActivationType(l, layers);
      NN_RETURN_STATUS();
    }

    if (l.getType() != "output") {
      status = realizeMultiOutputType(l, layers);
      NN_RETURN_STATUS();
    }

    if (l.getFlatten()) {
      status = realizeFlattenType(l, layers);
      NN_RETURN_STATUS();
    }
  }

  addLossLayer(loss_type);

  // std::list<LayerNode>::iterator iter;
  // for (unsigned int i = 0; i < adj.size(); ++i) {
  //   iter = adj[i].begin();
  //   for (unsigned int j = 0; j < (*iter).layer->input_layers.size(); ++j)
  //     std::cout << "      " << (*iter).layer->input_layers[j] << std::endl;
  //   std::cout << (*iter).index << " : " << (*iter).layer->getName()
  //             << std::endl;
  // }

  return status;
}

void NetworkGraph::setNumNetBufferSize() {
  for (unsigned int i = 0; i < Sorted.size(); ++i) {
    Sorted[i].layer->net_input.resize(Sorted[i].layer->input_layers.size());
    Sorted[i].layer->net_hidden.resize(Sorted[i].layer->output_layers.size());
  }
}

LayerNode &NetworkGraph::getLayerNode(const std::string &layer_name) {

  std::list<LayerNode>::iterator iter;
  for (unsigned int i = 0; i < adj.size(); ++i) {
    iter = adj[i].begin();
    if ((*iter).layer->getName() == layer_name)
      return (*iter);
  }

  throw std::invalid_argument("Cannot find Layer");
}

LayerNode &NetworkGraph::getSortedLayerNode(const std::string &layer_name) {

  for (unsigned int i = 0; i < Sorted.size(); ++i) {
    if (Sorted[i].layer->getName() == layer_name)
      return Sorted[i];
  }

  throw std::invalid_argument("Cannot find Layer");
}

int NetworkGraph::setEdge() {
  int status = ML_ERROR_NONE;

  std::list<LayerNode>::iterator iter;
  for (unsigned int i = 0; i < adj.size(); ++i) {
    iter = adj[i].begin();
    if ((*iter).layer->getType() == "input")
      continue;

    for (unsigned int j = 0; j < (*iter).layer->input_layers.size(); ++j) {
      unsigned int to_node_id =
        getLayerNode((*iter).layer->input_layers[j]).index;
      std::cout << getLayerNode(to_node_id).layer->getName() << " : "
                << (*iter).layer->getName() << " ";
      addEdge(to_node_id, (*iter));
    }
    std::cout << std::endl;
  }

  for (unsigned int i = 0; i < adj.size(); ++i) {

    std::list<LayerNode>::iterator iter;
    std::cout << i << " : " << getLayerNode(i).layer->getName() << " ("
              << getLayerNode(i).layer->num_inputs << " ";
    for (unsigned int j = 0; j < getLayerNode(i).layer->input_layers.size();
         ++j)
      std::cout << getLayerNode(i).layer->input_layers[j] << ", ";

    std::cout << " --> " << getLayerNode(i).layer->num_outputs << " ";
    for (unsigned int j = 0; j < getLayerNode(i).layer->output_layers.size();
         ++j)
      std::cout << getLayerNode(i).layer->output_layers[j] << ", ";

    std::cout << " )" << std::endl;

    for (iter = std::next(adj[i].begin()); iter != adj[i].end(); ++iter) {
      std::cout << "       " << (*iter).layer->getName();
    }
    std::cout << std::endl;
  }

  return status;
}

void NetworkGraph::setBatchSize(unsigned int batch_size) {
  for (auto const &layer_node : Sorted) {
    layer_node.layer->setBatch(batch_size);
  }
}

sharedConstTensors NetworkGraph::forwarding(sharedConstTensors input) {
  for (unsigned int i = 0; i < Sorted.size() - 1; ++i) {
    LayerNode &layer_node = Sorted[i];
    if (layer_node.layer->getType() == LayerType::LAYER_IN) {
      layer_node.layer->forwarding(input);
    } else {
      layer_node.layer->forwarding();
    }
  }

  std::vector<sharedConstTensor> out;

  for (unsigned int i = 0; i < Sorted[Sorted.size() - 2].layer->num_outputs;
       ++i) {
    out.push_back(
      MAKE_SHARED_TENSOR(Sorted[Sorted.size() - 2].layer->net_hidden[i]->var));
  }

  return out;
}

void NetworkGraph::backwarding(sharedConstTensors output, int iteration) {

  for (unsigned int i = Sorted.size() - 1; i > 0; i--) {
    LayerNode &layer_node = Sorted[i];
    if (layer_node.layer->getType() == LayerType::LAYER_LOSS) {
      layer_node.layer->backwarding(iteration, output);
    } else {
      layer_node.layer->backwarding(iteration);
    }
  }
}

} /* namespace nntrainer */
