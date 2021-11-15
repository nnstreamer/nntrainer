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
#include <profiler.h>
#include <rnn.h>
#include <split_layer.h>
#include <time_dist.h>
#include <util_func.h>

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

  setExecutionOrder();

  inPlaceOptimize();

  status = checkCompiledGraph();
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

void NetworkGraph::setExecutionOrder() {
  auto max_count = graph.size() * 3;
  /** @todo: remove backwarding count for non-trainble layers */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto &node = *iter;
    auto order_idx = iter - cbegin();
    auto forward_order = order_idx;
    auto calc_gradient_order = max_count - ((order_idx + 1) * 2);
    /** calc derivative is called right after calc_gradient */
    auto calc_derivative_order = calc_gradient_order + 1;
    node->setExecutionOrder(
      {forward_order, calc_gradient_order, calc_derivative_order});
  }
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
    if (layer->getNumInputConnections() == 0 &&
        !layer->hasInputShapeProperty()) {
      ml_logd("default input added %s->%s", prev_layer->getName().c_str(),
              layer->getName().c_str());
      layer->addInputLayers(prev_layer->getName());
    }
  }
}

void NetworkGraph::addLayerNode(std::unique_ptr<Layer> layer) {
  graph.addNode(std::make_unique<LayerNode>(std::move(layer)));
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

  props::Activation act_prop;
  act_prop.set(act);
  lnode->setProperty({"activation=" + to_string(act_prop)});
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
    if (lnode->getNumInputConnections() == 0) {
      if (!lnode->hasInputShapeProperty()) {
        ml_loge("Layer with no inbound connection need input_shape property");
        return ML_ERROR_INVALID_PARAMETER;
      }
    }
  }

  return ML_ERROR_NONE;
}

void NetworkGraph::markNodesForBackwarding() {
  /** accumulate all the nodes which must support backwarding */
  std::unordered_set<std::string> must_support_backwarding;

  /**
   * if a node is trainable, then all the nodes ahead of it must support
   * backwarding operation
   */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto lnode = (*iter);
    if (lnode->getTrainable() ||
        must_support_backwarding.find(lnode->getName()) !=
          must_support_backwarding.end()) {
      lnode->needsCalcGradient(true);
#ifdef ENABLE_TEST
      if (lnode->supportBackwarding() && !optimize_memory) {
        lnode->needsCalcDerivative(true);
      }
#endif
      for (auto const &out_layer : lnode->getOutputLayers()) {
        must_support_backwarding.insert(out_layer);
      }
    }
  }

  /** mark all the required nodes support backwarding */
  for (auto const &node_name : must_support_backwarding)
    LNODE(graph.getNode(node_name))->needsCalcDerivative(true);
}

int NetworkGraph::realizeGraph() {
  int status = ML_ERROR_NONE;

  addDefaultInputLayers();

  /**
   * invariant: the new realized nodes are added to the end,
   * otherwise this iteration becomes invalid. So, every iteration must be
   * fresh iterator as vector resize invalidates all the iterators.
   */
  for (unsigned int i = 0; i < graph.size(); ++i) {
    auto const &lnode = LNODE(*(cbegin() + i));
    ml_logd("layer name: %s", lnode->getName().c_str());

    /** If a layer does not has input nodes, then it must have input dimension
     */
    if (lnode->getNumInputConnections() == 0) {
      if (!lnode->hasInputShapeProperty()) {
        ml_loge("Input Dimension must be set");
        status = ML_ERROR_INVALID_PARAMETER;
        NN_RETURN_STATUS();
      }
    }

    if (lnode->getType() != ActivationLayer::type) {
      status = realizeActivationType(lnode);
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
   * otherwise this iteration becomes invalid. So, every iteration must be
   * fresh iterator as vector resize invalidates all the iterators.
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
  if (batch_size == this->batch_size)
    return;

  this->batch_size = batch_size;
  if (!input_list.empty() && getInputDimension()[0].batch() == batch_size)
    return;

  auto allocated = tensor_manager->isAllocated();

  if (allocated)
    deallocateTensors();

  for (auto iter = cbegin(); iter != cend(); iter++) {
    (*iter)->setBatch(batch_size);
    if ((*iter)->isFinalized()) {
      const RunLayerContext &context = (*iter)->getRunContext();
      // resize tensors spec
      for (unsigned int idx = 0; idx < context.getNumTensors(); idx++) {
        auto const &ts = context.getTensor(idx);
        tensor_manager->setBatchSize(ts.getName(), ts.getDim().batch());
        if (context.tensorHasGradient(idx)) {
          auto const &ts_grad = context.getTensorGrad(idx);
          tensor_manager->setBatchSize(ts_grad.getName(),
                                       ts_grad.getDim().batch());
        }
      }
    }
  }
  /// resize input and output spec
  tensor_manager->setBatchSize(batch_size);

  if (allocated)
    allocateTensors(exec_mode);

  /** update input and label dimensions */
  for (unsigned int idx = 0; idx < input_list.size(); idx++)
    input_dims[idx] = tensor_manager->getTensor(input_list[idx])->getDim();
  for (unsigned int idx = 0; idx < label_list.size(); idx++)
    label_dims[idx] = tensor_manager->getTensor(label_list[idx])->getDim();
}

void NetworkGraph::applyGradientsOnLastAccess(
  LayerNode *node, std::function<void(Weight &)> apply_func) {
  auto &rc = node->getRunContext();
  auto num_weight = rc.getNumWeights();
  for (unsigned i = 0; i < num_weight; ++i) {
    if (!rc.weightHasGradient(i)) {
      continue;
    }

    if (!rc.isGradientLastAccess(i)) {
      /// @note instead of checking the last access of the weight, checking
      /// if weights are dependent to others to minimize overhead.
      /// this logic assums that the source of the dependent weight must be
      /// prior to the dependent.
      continue;
    }

    apply_func(rc.getWeightObject(i));
  }
}

sharedConstTensors NetworkGraph::forwarding(bool training) const {
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto const &ln = *iter;
    START_PROFILE(profile_keys.at(ln->getType()));
    ln->forwarding(training);
    END_PROFILE(profile_keys.at(ln->getType()));
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

void NetworkGraph::backwarding(
  int iteration,
  std::function<void(std::shared_ptr<LayerNode>, int)> &backwarding_op) const {
  /**
   * last layer backwarding is run out of this loop
   */
  auto iter_begin = getBackwardingBeginIter();
  auto iter_end = getBackwardingEndIter();

  /// there is no layer to train, so backwarding is essentially noop
  if (iter_begin == iter_end) {
    return;
  }

  auto const &lptr_begin = (*iter_begin);

  if (lptr_begin->requireLabel() == false)
    throw std::runtime_error(
      "Error: last layer does not accept label, we can't train");

  for (auto iter = iter_begin; iter != iter_end; iter++) {
    auto &ln = *iter;
    START_PROFILE(profile_keys.at(ln->getType()));
    backwarding_op(ln, iteration);
    END_PROFILE(profile_keys.at(ln->getType()));
  }
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void NetworkGraph::allocateTensors(ExecutionMode exec_mode_) {
  exec_mode = exec_mode_;
  if (exec_mode == ExecutionMode::INFERENCE)
    /**
     * get the order of execution/usage order for the forwarding of the last
     * layer and pass that as the max_exec_order ensuring that all tensors
     * with usage less than the max_exec_order are allocated.
     */
    tensor_manager->allocateTensors(
      std::get<0>((*(cend() - 1))->getExecutionOrder()));
  else {
    /**
     * get the order of execution/usage order for the backwarding of the first
     * layer (as that will be the last layer to executed in the backwarding)
     * and pass that as the max_exec_order ensuring that all tensors with
     * usage less than the max_exec_order are allocated.
     */
    unsigned int max_exec_order = 0;
    if (!optimize_memory)
      max_exec_order = std::get<2>((*(cbegin()))->getExecutionOrder());
    for (auto iter = getBackwardingBeginIter(); iter != getBackwardingEndIter();
         iter++) {
      auto &ln = *iter;
      if (ln->needsCalcDerivative() || ln->needsCalcGradient()) {
#ifdef ENABLE_TEST
        max_exec_order =
          std::max(max_exec_order, std::get<2>(ln->getExecutionOrder()));
#else
        max_exec_order =
          std::max(max_exec_order, std::get<1>(ln->getExecutionOrder()));
#endif
      } else {
        max_exec_order =
          std::max(max_exec_order, std::get<0>(ln->getExecutionOrder()));
      }
    }
    tensor_manager->allocateTensors(max_exec_order);
  }
}

std::vector<TensorDim> NetworkGraph::getInputDimension() const {
  NNTR_THROW_IF(input_dims.empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node identified as input!";
  return input_dims;
}

unsigned int NetworkGraph::getBatchSize() const { return batch_size; }

std::vector<TensorDim> NetworkGraph::getOutputDimension() const {
  NNTR_THROW_IF(label_dims.empty(), std::invalid_argument)
    << "[NetworkGraph] the graph has no node identified as output!";
  /// for now, outputting label_dims works, later label dim will be different
  /// from output dimension
  return label_dims;
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
   * ini file and is overwritten here by the model loader for connection
   * making.
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

  /** @todo Update shared_from node name as well */
  /** @todo Add test for this */
}

void NetworkGraph::addLayer(std::shared_ptr<LayerNode> layer) {
  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /** Insert the layer to the graph */
  graph.addNode(layer);
}

InPlace
NetworkGraph::canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode) {
  if (!lnode->supportInPlace())
    return InPlace::NONE;

  /** layers which behave as a no-op - flatten */
  auto no_op = [](const std::shared_ptr<LayerNode> &lnode) {
    return lnode->getType() == FlattenLayer::type;
  };

  /** layers which behave as a no-op but shares memory among parallel nodes -
   * multiout */
  auto no_op_shared = [](const std::shared_ptr<LayerNode> &lnode) {
    return lnode->getType() == MultiOutLayer::type;
  };

  /**
   * layers whose backwarding is not dependent on input/output but only its
   * derivatives and weights, if any - batch normalization
   */
  auto io_independent_backwarding =
    [](const std::shared_ptr<LayerNode> &lnode) {
      return lnode->getType() == BatchNormalizationLayer::type;
    };

  /**
   * @note Conditions to decide if this layer node can be in-place:
   * 1. if the layer is a no-op, then it can operate in-place as it is not
   * modifying its input/output tensors and does not need to check its
   * neighboring nodes for dependency.
   * 2. if the layer is not supporting backwarding, there is no dependency
   * requirement with other nodes for backwarding.
   *
   * @note Conditions to decide the type of inplace for this layer:
   * 1. if the previous layers were restricting, then this layer will also be
   * restricting.
   * 2. if the previous layer were non_restricting or not inplace, then this
   * layer will be non-restricting.
   */
  if (no_op(lnode) || !lnode->supportBackwarding()) {
    auto const &input_layers = lnode->getInputLayers();
    for (unsigned int i = 0; i < input_layers.size(); ++i) {
      if (getLayerNode(input_layers[i])->executeInPlace() ==
          InPlace::RESTRICTING)
        return InPlace::RESTRICTING;
    }
    return InPlace::NON_RESTRICTING;
  }

  /**
   * @note Conditions to decide if this layer node can be in-place:
   * if the layer is a no-op-shared, then it can operate in-place as it is not
   * modifying its input/output tensors and does not need to check its
   * neighboring nodes for dependency.
   *
   * @note Conditions to decide the type of inplace for this layer:
   * As all the output nodes are sharing memory, the output nodes cant execute
   * inplace, and then its restricting mode.
   */
  if (no_op_shared(lnode))
    return InPlace::RESTRICTING;

  /**
   * @note Conditions to decide if this layer node can be in-place:
   * This is a generic case where the layer can support in-place but will modify
   * its input in-place. This includes layers like activation, etc. Apply checks
   * below to ensure that the layers can work in-place:
   * - if any of the input layer are restriction, then this layer cannot work
   *   as layers behind this layer have added restrictions.
   * - if all of the input layers are either not inplace or have no
   * restrictions, then this layer can operate in-place.
   *
   * @note Conditions to decide the type of inplace for this layer:
   * This is a generic case, and always restrictions on the next nodes to be not
   * inplace.
   *
   * @note This logic is prone to change as more layers are allowed to
   * work in-place such as concat layer, split layer, addition layer, dropout
   * layer, etc.
   *
   * @todo This logic sets layers to in-place one-by-one as they arrive. However
   * setting some layers to in-place can save more memory than others (like
   * multiout layer vs activaiton layer). The layers need to sorted based on the
   * memory save they provide and then make them in-place in that order.
   */
  if (lnode->getType() == ActivationLayer::type ||
      lnode->getType() == BatchNormalizationLayer::type) {
    auto const &input_layers = lnode->getInputLayers();
    for (unsigned int i = 0; i < input_layers.size(); ++i) {
      if (getLayerNode(input_layers[i])->executeInPlace() ==
          InPlace::RESTRICTING)
        return InPlace::NONE;
    }

    /**
     * if the layer does io_independent_backwarding where the input and output
     * is not requried during backwarding, then it is a non-restricting in-place
     * layer.
     */
    if (io_independent_backwarding(lnode))
      return InPlace::NON_RESTRICTING;

    return InPlace::RESTRICTING;
  }

  return InPlace::NONE;
}

void NetworkGraph::inPlaceOptimize() {
  if (optimize_memory) {
    for (unsigned int idx = 0; idx < graph.size(); ++idx) {
      auto const &lnode = getSortedLayerNode(idx);
      lnode->executeInPlace(canExecuteInPlace(lnode));
    }
  }
}

/**
 * @brief Set the Inplace Shared Memory Config By Layer object
 *
 * @param lnode layer node object
 * @param shared_var if the variable should be shared
 * @param shared_grad if the gradient should be shared
 */
static void
setInplaceSharedMemoryConfigByLayer(const std::shared_ptr<LayerNode> &lnode,
                                    bool &shared_var, bool &shared_grad) {
  /** for multiout layer, variables are shared but gradients are not */
  if (lnode->getType() == MultiOutLayer::type) {
    shared_var = true;
    shared_grad = false;
  } else {
    shared_var = true;
    shared_grad = true;
  }
  /** @todo for addition layer, variables are not shared but gradients are */
  /**
   * @todo for layers which support in-place, both variables and gradients will
   * be be shared.
   *
   * @todo add a check here is the layer being checked here can support in-place
   * or not
   */
}

std::vector<Var_Grad *>
NetworkGraph::finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                              const std::vector<Var_Grad *> &prev_inputs) {
  const GraphNode &gnode = *lnode.get();
  std::vector<TensorDim> input_dims;
  input_dims.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_dims),
                 [](const Var_Grad *vg) { return vg->getDim(); });

  /** finalize the layer and get the final context */
  auto init_context = lnode->finalize(input_dims);

  /**
   * Request manager for either a pre-allocated output as input or a newly
   * allocated input. This is necesary for manager to know when this input node
   * is going to be used.
   */
  std::vector<std::string> input_names;
  input_names.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_names),
                 [](auto const &vg) { return vg->getName(); });
  const std::vector<Var_Grad *> &inputs = tensor_manager->requestInputs(
    gnode, init_context.getInputDimensions(), input_names);

  /** In-Place optimizations */
  std::vector<std::string> inputs_name;
  bool shared_var = false, shared_grad = false;
  if (lnode->executeInPlace() != InPlace::NONE) {
    std::transform(inputs.begin(), inputs.end(),
                   std::back_inserter(inputs_name),
                   [](const Var_Grad *val) { return val->getName(); });
    setInplaceSharedMemoryConfigByLayer(lnode, shared_var, shared_grad);
  }

  /**
   * Request manager for either a pre-allocated input as output or a newly
   * allocated input. This is necesary for manager to know when this output node
   * is going to be used with in-place optimizations.
   */
  const std::vector<Var_Grad *> &outputs =
    tensor_manager->requestOutputs(gnode, init_context.getOutputDimensions(),
                                   inputs_name, shared_var, shared_grad);

  /** create shared weight names if requested */
  std::vector<std::string> shared_weight_names;
  std::vector<std::string> shared_tensor_names;
  if (auto shared_node_str = lnode->getSharedFrom(); !shared_node_str.empty()) {
    /// @note below is commented but kept from quick fix to be referenced for
    /// later(#1707)
    // auto shared_node = getLayerNode(shared_node_str).get();
    // NNTR_THROW_IF(shared_node == nullptr, std::invalid_argument)
    //   << "shared_node requested but it is not registered in the graph, name:
    //   "
    //   << shared_node_str << " requested from " << lnode->getName();
    // NNTR_THROW_IF(shared_node->getType() != lnode->getType(),
    //               std::invalid_argument)
    //   << " shared_node and lnode type mismatch, source node type: "
    //   << shared_node->getType() << " depedent node type: " <<
    //   lnode->getType()
    //   << " depedent node name: " << lnode->getName();
    // NNTR_THROW_IF(!shared_node->isFinalized(), std::invalid_argument)
    //   << "shared node must be prior to the dependent node and it should be "
    //      "finalized beforehand, shared node name: "
    //   << shared_node_str << " dependent node name: " << lnode->getName();
    // auto num_weight = shared_node->getNumWeights();
    // shared_weight_names.reserve(num_weight);
    // for (auto i = 0u; i < num_weight; ++i) {
    //   shared_weight_names.emplace_back(shared_node->getWeightName(i));
    // }
    // auto &rc = node->getRunContext();

    /// @fixme tensor should be only shared if context explicitly requested to
    /// do so. This has to be added to the part of tensor spec, other wise it
    /// will break many things
    const auto &t_specs = init_context.getTensorsSpec();
    for (auto i = 0u; i < t_specs.size(); ++i) {
      shared_tensor_names.emplace_back(std::get<3>(t_specs.at(i)));
    }

    const auto &w_specs = init_context.getWeightsSpec();
    for (auto i = 0u; i < w_specs.size(); ++i) {
      shared_weight_names.emplace_back(std::get<5>(w_specs.at(i)));
    }
  }

  lnode->configureRunContext(
    // TODO: update weights spec for trainable based on layer trainable prop
    tensor_manager->requestWeights(gnode, init_context.getWeightsSpec(),
                                   lnode->getTrainable(), shared_weight_names),
    inputs, outputs,
    tensor_manager->requestTensors(gnode, init_context.getTensorsSpec(),
                                   shared_tensor_names));

  return outputs;
}

int NetworkGraph::initialize(
  const std::vector<std::string> &model_input_names,
  const std::vector<std::string> &model_label_names) {

  /**
   * this contains the map from node name to its input tensor names
   * @note: these input tensors have already been allocated
   */
  std::unordered_map<std::string, std::vector<Var_Grad *>> input_map;

  /** check if the given config of node is of input node */
  auto is_input_node = [](const LayerNode *node) -> bool {
    return node->getInputConnections().empty();
  };

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    std::vector<Var_Grad *> inputs = {};
    auto const &lnode = getSortedLayerNode(idx);
    ml_logd("layer name : %s", lnode->getName().c_str());

    if (profile_keys.find(lnode->getType()) == profile_keys.end()) {
      int event_key = 0;
      REGISTER_EVENT(lnode->getType(), event_key);
      profile_keys[lnode->getType()] = event_key;
    }

    /**
     * Set input dimension for all the layers.
     * For input layer, as input dimension is known, set input tensor.
     */
    if (!is_input_node(lnode.get())) {
      if (input_map.find(lnode->getName()) == input_map.end())
        throw std::runtime_error("Cannot find input buffers for the node");
      inputs = input_map.at(lnode->getName());
    }

    /**
     * Initialize all the layers, allocate output tensors for each layer
     * init2and add optimizer related weights for the layer
     */
    const std::vector<Var_Grad *> &outputs = finalizeContext(lnode, inputs);

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

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    auto const &lnode = getSortedLayerNode(idx);
    auto &rc = lnode->getRunContext();
    auto first_grad_access = std::get<1>(lnode->getExecutionOrder());
    auto last_grad_access = std::get<2>(lnode->getExecutionOrder());
    for (unsigned i = 0; i < rc.getNumWeights(); ++i) {
      if (!rc.weightHasGradient(i)) {
        /// @todo this is duck taping that MUST BE REMOVED. We will need to
        /// have, is weight first access kind of concept.
        if (tensor_manager->isFirstAccess(
              rc.getWeight(i).getName(),
              std::get<0>(lnode->getExecutionOrder()), true)) {
          rc.getWeightObject(i).setAsGradientFirstAccess();
        }
        if (tensor_manager->isLastAccess(rc.getWeight(i).getName(),
                                         last_grad_access, true)) {
          rc.getWeightObject(i).setAsGradientLastAccess();
        }
      } else {
        if (tensor_manager->isFirstAccess(rc.getWeightGrad(i).getName(),
                                          first_grad_access)) {
          rc.getWeightObject(i).setAsGradientFirstAccess();
        }
        if (tensor_manager->isLastAccess(rc.getWeightGrad(i).getName(),
                                         last_grad_access)) {
          rc.getWeightObject(i).setAsGradientLastAccess();
        }
      }
    }
  }
  /**** identify model input / output to be set externally later ****/
  auto identify_as_model_input = [this](LayerNode *node) {
    auto num_input = node->getNumInputs();
    NNTR_THROW_IF(num_input != 1, std::invalid_argument)
      << "Input layer is supposed to have exactly one input, but more then "
         "one input detected, num inputs: "
      << num_input;

    input_list.push_back(node->getInput(0).getName());
    input_dims.push_back(node->getInputDimensions()[0]);
  };

  auto is_label_node = [](LayerNode *node) { return node->requireLabel(); };

  auto identify_as_model_label = [this](LayerNode *node) {
    /// @todo change this as lnode->getNumLabels of sorts
    auto num_label = node->getNumOutputs();
    NNTR_THROW_IF(!node->getOutputConnections().empty(), std::invalid_argument)
      << "label layer is supposed to be a leaf for now";
    NNTR_THROW_IF(num_label != 1, std::invalid_argument)
      << "label layer is supposed to have exactly one label, but more then "
         "one label detected, num labels: "
      << num_label;

    /// @todo implement and use getLabel(0) instead.
    output_list.push_back(node->getOutput(0).getName());
    label_list.push_back(node->getOutputGrad(0).getName());
    label_dims.push_back(node->getOutputDimensions()[0]);
  };

  auto identify_external_tensors = [this](const std::vector<std::string> &names,
                                          auto &&pred, auto &&identify) {
    if (names.empty()) {
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        /// when name is empty, we identify everything as the node, all of them
        /// must be having identical dimensions
        identify(lnode);
      }
    } else {
      for (auto &name : names) {
        auto lnode = getLayerNode(name).get();
        NNTR_THROW_IF(!pred(lnode), std::invalid_argument)
          << "given node is not of that kind, name: " << name;
        identify(lnode);
      }
      unsigned int num_node_of_kind = 0;
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        num_node_of_kind++;
      }
      NNTR_THROW_IF(num_node_of_kind != names.size(), std::invalid_argument)
        << "names given but there are not identified node of the kind, num "
           "node of kind: "
        << num_node_of_kind << " identifier size: " << names.size();
    }
  };

  identify_external_tensors(model_input_names, is_input_node,
                            identify_as_model_input);
  identify_external_tensors(model_label_names, is_label_node,
                            identify_as_model_label);

  /** mark the nodes which will be backwarded during the graph operation */
  try {
    markNodesForBackwarding();
  } catch (std::exception &e) {
    ml_loge(
      "Backwarding required from layer which doesn't support backwarding: %s",
      e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

void NetworkGraph::setExternalTensors(const std::vector<Tensor> &data,
                                      const std::vector<std::string> names) {

  /// feed or clear label
  for (unsigned int idx = 0; idx < names.size(); idx++) {
    if (data.empty())
      tensor_manager->fillPlaceholder(names[idx], Tensor());
    else if (data.size() == 1)
      tensor_manager->fillPlaceholder(names[idx], data[0]);
    else
      tensor_manager->fillPlaceholder(names[idx], data[idx]);
  }
}

void NetworkGraph::setInputsLabels(const std::vector<Tensor> &inputs,
                                   const std::vector<Tensor> &labels) {

  NNTR_THROW_IF(labels.size() > 1 && labels.size() != label_list.size(),
                std::invalid_argument)
    << "label size does not match with the network requirements"
    << " label size: " << labels.size()
    << " requirements size: " << label_list.size();

  NNTR_THROW_IF(inputs.size() > 1 && inputs.size() != input_list.size(),
                std::invalid_argument)
    << "input size does not match with the network requirements"
    << " input size: " << inputs.size()
    << " requirements size: " << input_list.size();

  setExternalTensors(inputs, input_list);
  setExternalTensors(labels, label_list);
}

void NetworkGraph::setInputsLabels(sharedConstTensors &inputs,
                                   sharedConstTensors &labels) {

  std::vector<Tensor> ins;
  std::transform(inputs.begin(), inputs.end(), std::back_inserter(ins),
                 [](auto const &val) { return *val.get(); });

  std::vector<Tensor> labs;
  std::transform(labels.begin(), labels.end(), std::back_inserter(labs),
                 [](auto const &val) { return *val.get(); });

  setInputsLabels(ins, labs);
}

std::vector<Tensor> NetworkGraph::getOutputTensors() const {
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(output_list.size());

  for (auto const &name : output_list)
    output_tensors.push_back(*tensor_manager->getTensor(name));

  return output_tensors;
}

} /* namespace nntrainer */
