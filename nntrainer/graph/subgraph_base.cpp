// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    subgraph_base.cpp
 * @date    07 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network SubGraph Class for Neural Network
 *
 * @todo    Support multi-input graph.
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include <activation_layer.h>
#include <cross_entropy_loss_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>

#include <input_layer.h>
#include <layer_node.h>
#include <multiout_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <optimizer_context.h>
#include <profiler.h>
#include <util_func.h>

#include <subgraph_base.h>

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

void SubGraphBase::setProperty(const std::vector<std::string> &properties) {
  auto left_properties = loadProperties(properties, *subgraph_props);
}

const std::string SubGraphBase::getName() const noexcept {
  auto &name = std::get<props::SubGraphName>(*subgraph_props);
  return name.empty() ? "default" : name.get();
}

void SubGraphBase::setName(const std::string &name) {
  setProperty({"subgraph_name=" + name});
}

void SubGraphBase::finalize() {

  /** finalize properties */
  subgraph_name = getName();
}

const std::string SubGraphBase::getType() const { return SubGraphBase::type; }

bool SubGraphBase::getTrainable() const {
  return exec_mode == ExecutionMode::TRAIN;
}

const std::vector<std::string> SubGraphBase::getInputConnections() const {
  return cbegin()->getInputConnections();
}

const std::vector<std::string> SubGraphBase::getOutputConnections() const {
  return crbegin()->getOutputConnections();
}

ExecutionOrder SubGraphBase::getExecutionOrder() const { return exec_order; }

void SubGraphBase::setExecutionOrder(ExecutionOrder exec_order_) {
  exec_order = exec_order_;
}

void SubGraphBase::setExecutionOrder() {
  auto backward_order = subgraph.size();
  for (auto iter = getBackwardingBeginIter(); iter != getBackwardingEndIter();
       iter++) {
    auto &node = *iter;
    auto order_idx = getBackwardingEndIter() - iter - 1;
    auto forward_order = order_idx;
    auto calc_gradient_order = backward_order;
    if (node->getTrainable())
      backward_order++;
    auto calc_derivative_order = backward_order;
    if (node->getTrainable())
      backward_order++;
    auto apply_gradient_order = backward_order++;

    node->setExecutionOrder({forward_order, calc_gradient_order,
                             calc_derivative_order, apply_gradient_order});
  }

  /**
   * This sets max execution order temporarily till model is initialized.
   * This set max execution order is used to extend gradient exec orders for
   * clipping.
   */
  graph_exec_end = std::get<3>((*(cbegin()))->getExecutionOrder());
}

int SubGraphBase::addLossLayer(const std::string &loss_type_) {
  for (unsigned int i = 0; i < subgraph.getNumOutputNodes(); ++i) {
    auto output_layer_node = LNODE(subgraph.getOutputNode(i));
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
        LNODE(subgraph.getNode(output_layer_node->getInputConnectionName(0)));
    }

    std::shared_ptr<LayerNode> lnode = createLayerNode(loss_type);
    subgraph.ensureName(*lnode);

    if (second_to_last_layer_node->getDistribute()) {
      lnode->setProperty({"distribute=true"});
    }

    /// @todo remove this by add loss at realization
    second_to_last_layer_node->setOutputLayers({lnode->getName()});
    lnode->setProperty(
      {"input_layers=" + second_to_last_layer_node->getName()});

    if (is_cross_entropy_loss) {
      subgraph.replaceNode(output_layer_node, lnode);
    } else {
      subgraph.addNode(lnode, false);
    }
    subgraph.replaceOutputNode(i, lnode);
  }

  return ML_ERROR_NONE;
}

void SubGraphBase::setOutputConnections() {
  for (auto layer_iter = cbegin(); layer_iter != cend(); ++layer_iter) {
    const auto &node = *layer_iter;
    for (auto i = 0u, num_inode = node->getNumInputConnections(); i < num_inode;
         ++i) {
      const auto &name = node->getInputConnectionName(i);
      const auto &idx = node->getInputConnectionIndex(i);

      auto node_setting_output = getLayerNode(name);
      node_setting_output->setOutputConnection(idx, node->getName(), i);
    }
  }
}

int SubGraphBase::isCompilable() {
  if (compiled) {
    ml_loge("SubGraph is already compiled");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (subgraph.empty()) {
    ml_loge("SubGraph is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

int SubGraphBase::checkCompiledGraph() {
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

void SubGraphBase::markNodesForBackwarding() {
  /** accumulate all the nodes which must support backwarding */
  std::unordered_set<std::string> must_support_backwarding;
  if (exec_mode == ExecutionMode::INFERENCE) {
    for (auto iter = cbegin(); iter != cend(); iter++) {
      auto lnode = (*iter);
      lnode->needsCalcGradient(false);
      lnode->needsCalcDerivative(false);
    }
    return;
  }

  /**
   * if a node is trainable, then all the nodes ahead of it must support
   * backwarding operation
   */
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto lnode = (*iter);
    if (lnode->getTrainable() ||
        must_support_backwarding.find(lnode->getName()) !=
          must_support_backwarding.end()) {
      if (lnode->getTrainable()) {
        lnode->needsCalcGradient(true);
      }
#ifdef ENABLE_TEST
      if (lnode->supportBackwarding() && !optimize_memory) {
        lnode->needsCalcDerivative(true);
      }
#endif

      for (auto i = 0u, num_node = lnode->getNumOutputConnections();
           i < num_node; ++i) {
        auto conn = lnode->getOutputConnection(i);
        if (!conn) {
          continue;
        }

        must_support_backwarding.insert(conn->getName());
      }
    }
  }

  /** mark all the required nodes support backwarding */
  for (auto const &node_name : must_support_backwarding) {
    auto ln = LNODE(subgraph.getNode(node_name)).get();
    ln->needsCalcDerivative(true);
  }
}

LayerNode *SubGraphBase::computeBackwardEnd() {
  int max_exec_order = -1;
  LayerNode *node = nullptr;

  if (!optimize_memory) {
    return (*cbegin()).get();
  }

  for (auto iter = getBackwardingBeginIter(); iter != getBackwardingEndIter();
       iter++) {
    auto &ln = *iter;
    const auto &exec_order = ln->getExecutionOrder();
    int cur_order = std::get<0>(exec_order);
    if (ln->needsCalcDerivative() || ln->needsCalcGradient()) {
#ifdef ENABLE_TEST
      cur_order = std::get<2>(exec_order);
#else
      cur_order = std::get<1>(exec_order);
#endif
    }

    NNTR_THROW_IF(max_exec_order == cur_order, std::invalid_argument)
      << "layer node: " << ln->getName()
      << " has duplicated max_exec_order, this should not happen, current "
         "execution order: "
      << max_exec_order;

    if (max_exec_order < cur_order) {
      max_exec_order = cur_order;
      node = ln.get();
    }
  }

  return node;
}

std::vector<TensorDim> SubGraphBase::getInputDimension() const {
  NNTR_THROW_IF(input_dims.empty(), std::invalid_argument)
    << "[SubGraphBase] the graph has no node identified as input!";
  return input_dims;
}

unsigned int SubGraphBase::getBatchSize() const { return batch_size; }

std::vector<TensorDim> SubGraphBase::getOutputDimension() const {
  NNTR_THROW_IF(label_dims.empty(), std::invalid_argument)
    << "[SubGraphBase] the graph has no node identified as output!";
  /// for now, outputting label_dims works, later label dim will be different
  /// from output dimension
  return label_dims;
}

std::vector<std::shared_ptr<LayerNode>>
SubGraphBase::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {
  /// @fixme: this won't work if input, output layers are not in order
  /// Further, this function must be removed. There should be rather
  /// getAllNames and getLayerByName instead of getUnsortedLayers.

  /** count layers after output layer */
  unsigned int num_layers_remove_end = 0;
  if (!output_layer.empty()) {
    for (auto iter = subgraph.crbegin(); iter != subgraph.crend(); iter++) {
      if ((*iter)->getName() != output_layer)
        num_layers_remove_end++;
      else
        break;
    }
  }

  if (num_layers_remove_end == subgraph.size())
    return {};

  /** count layers before input layer */
  unsigned int num_layers_remove_start = 0;
  if (!input_layer.empty()) {
    for (auto iter = subgraph.cbegin();
         iter != subgraph.cend() - num_layers_remove_end; iter++) {
      if ((*iter)->getName() != input_layer)
        num_layers_remove_start++;
      else
        break;
    }
  }

  /** copy the graph and return */
  std::vector<std::shared_ptr<LayerNode>> ret;
  std::transform(subgraph.cbegin() + num_layers_remove_start,
                 subgraph.cend() - num_layers_remove_end,
                 std::back_inserter(ret),
                 [](auto const &elem) { return LNODE(elem); });

  return ret;
}

std::vector<std::shared_ptr<LayerNode>> SubGraphBase::getLayerNodes() const {
  return std::vector<std::shared_ptr<LayerNode>>(cbegin(), cend());
}

void SubGraphBase::addLayer(std::shared_ptr<LayerNode> layer) {
  if (compiled)
    throw std::runtime_error("Cannot modify graph after compile");

  /** Insert the layer to the graph */
  subgraph.addNode(layer);
}

InPlaceType
SubGraphBase::canExecuteInPlace(const std::shared_ptr<LayerNode> &lnode) {
  InPlaceType inplace_type = lnode->initializeInPlace();

  if (inplace_type == InPlaceType::NONE) {
    return inplace_type;
  }

  if (lnode->getType() == InputLayer::type &&
      !istrequal(getTensorType()[2], "FP32")) {
    return InPlaceType::NONE;
  }

  if (lnode->getType() == MultiOutLayer::type) {
    return InPlaceType::RESTRICTING;
  }

  /** A case where it can operate in-place even if there is a multi-out type
   * input connection. */
  if (inplace_type == InPlaceType::RESTRICTING) {
    for (size_t i = 0, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      const std::string &input_name = lnode->getInputConnectionName(i);
      if (getLayerNode(input_name)->getInPlaceType() ==
          InPlaceType::RESTRICTING)
        return inplace_type;
    }
    return InPlaceType::NON_RESTRICTING;
  }
  /** A case where it cannot operate in-place if there is a multi-out type
   * input connection. */
  else { /** condition: NON_RESTRICTING */
    for (size_t i = 0, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      const std::string &input_name = lnode->getInputConnectionName(i);
      if (getLayerNode(input_name)->getInPlaceType() ==
          InPlaceType::RESTRICTING)
        return InPlaceType::NONE;
    }
    return inplace_type;
  }
}

void SubGraphBase::inPlaceOptimize() {
  if (optimize_memory) {
    for (unsigned int idx = 0; idx < subgraph.size(); ++idx) {
      auto const &lnode = getSortedLayerNode(idx);
      lnode->setInPlaceType(canExecuteInPlace(lnode));
    }
  }
}

#ifdef ENABLE_TEST

std::map<std::string, std::vector<unsigned int>>
SubGraphBase::getLayerExecutionOrders(const std::shared_ptr<LayerNode> &lnode) {
  const auto &init_context = lnode->getInitContext();
  auto out_specs = init_context.getOutSpecs();
  auto weight_specs = init_context.getWeightsSpec();
  auto tensor_specs = init_context.getTensorsSpec();

  std::map<std::string, std::vector<unsigned int>> exec_orders;

  for (auto &spec : out_specs) {
    const auto &name = lnode->getName() + ":" + spec.variable_spec.name;
    auto orders = tensor_manager->getTensorExecutionOrders(name, false);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  for (auto &spec : weight_specs) {
    const auto &name = std::get<const std::string>(spec);
    auto orders = tensor_manager->getTensorExecutionOrders(name, true);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  for (auto &spec : tensor_specs) {
    const auto &name = std::get<const std::string>(spec);
    auto orders = tensor_manager->getTensorExecutionOrders(name, false);
    exec_orders.insert({name, orders});
    try {
      auto orders_grad =
        tensor_manager->getTensorExecutionOrders(name + ":grad", false);
      exec_orders.insert({name + ":grad", orders_grad});
    } catch (const std::exception &e) {
      ml_logi("Cannot find grad tensor for %s:grad", name.c_str());
      continue;
    }
  }

  return exec_orders;
}

#endif // ENABLE_TEST

void SubGraphBase::setExternalTensors(const std::vector<Tensor> &data,
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

void SubGraphBase::setInputsLabels(const std::vector<Tensor> &inputs,
                                   const std::vector<Tensor> &labels) {
  setInputs(inputs);
  setLabels(labels);
}

void SubGraphBase::setInputsLabels(sharedConstTensors &inputs,
                                   sharedConstTensors &labels) {
  setInputs(inputs);
  setLabels(labels);
}

void SubGraphBase::setInputs(const std::vector<Tensor> &inputs) {
  NNTR_THROW_IF(inputs.size() > 1 && inputs.size() != input_list.size(),
                std::invalid_argument)
    << "input size does not match with the network requirements"
    << " input size: " << inputs.size()
    << " requirements size: " << input_list.size();
  setExternalTensors(inputs, input_list);
}

void SubGraphBase::setInputs(sharedConstTensors &inputs) {
  std::vector<Tensor> ins;
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(ins),
    [](auto const &val) -> const auto & { return *val.get(); });
  setInputs(ins);
}

void SubGraphBase::setLabels(const std::vector<Tensor> &labels) {
  NNTR_THROW_IF(labels.size() > 1 && labels.size() != label_list.size(),
                std::invalid_argument)
    << "label size does not match with the network requirements"
    << " label size: " << labels.size()
    << " requirements size: " << label_list.size();
  setExternalTensors(labels, label_list);
}

void SubGraphBase::setLabels(sharedConstTensors &labels) {
  std::vector<Tensor> labs;
  std::transform(
    labels.begin(), labels.end(), std::back_inserter(labs),
    [](auto const &val) -> const auto & { return *val.get(); });

  setLabels(labs);
}

std::vector<Tensor> SubGraphBase::getOutputTensors() const {
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(output_list.size());

  for (auto const &name : output_list)
    output_tensors.push_back(*tensor_manager->getTensor(name));

  return output_tensors;
}

void SubGraphBase::LoadTensors(unsigned int order) {
  tensor_manager->LoadTensors(order);
}

bool SubGraphBase::checkLoadComplete(unsigned int order) {
  return tensor_manager->checkLoadComplete(order);
}

bool SubGraphBase::checkUnloadComplete(unsigned int order) {
  return tensor_manager->checkUnloadComplete(order);
}

void SubGraphBase::UnloadTensors(unsigned int order) {
  tensor_manager->UnloadTensors(order);
}

void SubGraphBase::requestOptimizerVariable(
  std::function<std::vector<TensorDim>(const TensorDim &)> cb,
  bool request_only_trainable) {
  for (auto const &w : tensor_manager->getWeights()) {
    if (w->isGradientLastAccess() && w->hasGradient()) {
      const TensorDim &dim = w->getDim();
      std::vector<TensorDim> dims = cb(dim);
      w->setOptimizerVariables(tensor_manager->requestWeightOptimizerVariables(
        dims, w->getName(), ":opt", TensorLifespan::MAX_LIFESPAN,
        w->isGradientClipByGlobalNorm(), w->isMixedPrecision(),
        Initializer::ZEROS));
    }
  }
}

void SubGraphBase::resetLossScale(float scale) {
  loss_scale = scale;
  for (auto iter = cbegin(); iter != cend(); iter++) {
    auto &ln = *iter;
    ln->getRunContext().setLossScale(scale);
  }
}

void SubGraphBase::read(std::ifstream &file, bool opt_var,
                        ml::train::ExecutionMode mode) {
  for (auto layer_iter = cbegin(); layer_iter != cend(); ++layer_iter) {
    (*layer_iter)->read(file, opt_var, mode);
  }
}

void SubGraphBase::save(std::ofstream &file, bool opt_var,
                        ml::train::ExecutionMode mode) const {
  for (auto layer_iter = cbegin(); layer_iter != cend(); ++layer_iter) {
    (*layer_iter)->save(file, opt_var, mode);
  }
}

float SubGraphBase::getLoss() const {
  auto loss = 0.0f;

  for (auto layer_iter = cbegin(); layer_iter != cend(); ++layer_iter) {
    loss += (*layer_iter)->getLoss();
  }
  return loss;
}

void SubGraphBase::clearOptVar() {
  for (auto layer_iter = cbegin(); layer_iter != cend(); ++layer_iter) {
    (*layer_iter)->clearOptVar();
  }
}

void SubGraphBase::printPreset(std::ostream &out, PrintPreset preset) {
  for (auto layer_iter = cbegin(); layer_iter != cend(); ++layer_iter) {
    (*layer_iter)->printPreset(out, preset);
  }
}

} /* namespace nntrainer */
