// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    network_graph.h
 * @date    19 Oct 2020
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network Graph Class for Neural Network
 *
 * @todo    Support multi-input graph.
 */

#include <network_graph.h>
#include <optimizer_context.h>

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

int NetworkGraph::compile(const std::string &loss_type) {
  int status = ML_ERROR_NONE;
  for (auto it = cbegin(); it != cend(); ++it) {
    status = (*it)->compile(loss_type);
    if (status != ML_ERROR_NONE)
      return status;
  }
  return status;
}

void NetworkGraph::setBatchSize(unsigned int batch_size) {
  for (auto it = cbegin(); it != cend(); ++it) {
    (*it)->setBatchSize(batch_size);
  }
}

void NetworkGraph::applyGradients(LayerNode *node, int iteration,
                                  std::shared_ptr<OptimizerWrapped> opt) {
  for (auto it = cbegin(); it != cend(); ++it) {
    (*it)->applyGradients(node, iteration, opt);
  }
}

sharedConstTensors
NetworkGraph::forwarding(bool training,
                         std::function<bool(void *userdata)> stop_cb,
                         void *userdata, bool swap_mode) {
  sharedConstTensors output;
  for (auto it = cbegin(); it != cend(); ++it) {
    output = (*it)->forwarding(training, stop_cb, userdata, swap_mode);
  }
  return output;
}

sharedConstTensors NetworkGraph::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  sharedConstTensors output;
  for (auto it = cbegin(); it != cend(); ++it) {
    output =
      (*it)->incremental_forwarding(from, to, training, stop_cb, userdata);
  }
  return output;
}

bool NetworkGraph::backwarding(int iteration,
                               std::function<bool(void *userdata)> stop_cb,
                               void *user_data, bool is_grad_opt_mode,
                               std::shared_ptr<OptimizerWrapped> opt) {
  bool status = false;
  for (auto it = crbegin(); it != crend(); ++it) {
    status =
      (*it)->backwarding(iteration, stop_cb, user_data, is_grad_opt_mode, opt);
  }
  return status;
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void NetworkGraph::allocateTensors(ExecutionMode exec_mode_) {
  for (auto it = cbegin(); it != cend(); ++it) {
    (*it)->allocateTensors(exec_mode_);
  }
}

std::vector<TensorDim> NetworkGraph::getInputDimension() const {
  return (*cbegin())->getInputDimension();
}

unsigned int NetworkGraph::getBatchSize() const {
  return (*cbegin())->getBatchSize();
}

std::vector<TensorDim> NetworkGraph::getOutputDimension() const {
  return (*crbegin())->getOutputDimension();
}

std::vector<std::shared_ptr<LayerNode>>
NetworkGraph::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {
  std::vector<std::shared_ptr<LayerNode>> lns;
  for (auto it = cbegin(); it != cend(); ++it) {
    auto lns_ = (*it)->getUnsortedLayers(input_layer, output_layer);
    lns.insert(lns.end(), lns_.begin(), lns_.end());
  }
  return lns;
}

std::vector<std::shared_ptr<LayerNode>> NetworkGraph::getLayerNodes() const {
  std::vector<std::shared_ptr<LayerNode>> lns;
  for (auto it = cbegin(); it != cend(); ++it) {
    const auto &sg = (*it);
    auto lns_ = sg->getLayerNodes();
    lns.insert(lns.end(), lns_.begin(), lns_.end());
  }
  return lns;
}

void NetworkGraph::addLayer(std::shared_ptr<LayerNode> layer) {
  /**
   * @note This code written based on the assumption that he graph consists
   * with only one default subgraph node. It needs to be updated.
   * @todo it needs to verify the name of subgraph and add the layer to the
   * subgraph
   */
  const std::string &graph_name("default_subgraph");
  SGNODE(graph.getNode(graph_name))->addLayer(layer);
}

std::vector<Var_Grad *>
NetworkGraph::finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                              const std::vector<Var_Grad *> &prev_inputs) {
  /**
   * @note This code written based on the assumption that he graph consists
   * with only one default subgraph node. It needs to be updated.
   * @todo finalizeContext can be implemented at the network_graph not subgraph
   */
  return (*cbegin())->finalizeContext(lnode, prev_inputs);
}

std::vector<Var_Grad *>
NetworkGraph::refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                                const std::vector<Var_Grad *> &prev_inputs) {
  /**
   * @note This code written based on the assumption that he graph consists
   * with only one default subgraph node. It needs to be updated.
   * @todo refinalizeContext can be implemented at the network_graph not
   * subgraph
   */
  return (*cbegin())->refinalizeContext(lnode, prev_inputs);
}

#ifdef ENABLE_TEST

std::map<std::string, std::vector<unsigned int>>
NetworkGraph::getLayerExecutionOrders(const std::shared_ptr<LayerNode> &lnode) {
  /**
   * @note This code written based on the assumption that he graph consists
   * with only one default subgraph node. It needs to be updated.
   * @todo getLayerExecutionOrders can be implemented at the network_graph not
   * subgraph
   */
  return (*cbegin())->getLayerExecutionOrders(lnode);
}

#endif // ENABLE_TEST

int NetworkGraph::initialize(ExecutionMode mode,
                             const std::vector<Connection> &model_input_names,
                             const std::vector<Connection> &model_label_names) {

  /**
   * @note This code written based on the assumption that he graph consists
   * with only one default subgraph node. It needs to be updated.
   * @todo needs to verify the subgraph which requires external input/output.
   * Based on the info, the initialize should be updated.
   */
  return (*cbegin())->initialize(mode, model_input_names, model_label_names);
}

int NetworkGraph::reinitialize(
  const std::vector<Connection> &model_input_names,
  const std::vector<Connection> &model_label_names) {
  return (*cbegin())->reinitialize(model_input_names, model_label_names);
}

void NetworkGraph::setInputsLabels(const std::vector<Tensor> &inputs,
                                   const std::vector<Tensor> &labels) {
  /**
   * @note This code written based on the assumption that he graph consists
   * with only one subgraph requiring inputs and one subgraph requiring labels.
   * This should be updated later.
   */
  (*cbegin())->setInputs(inputs);
  (*crbegin())->setLabels(labels);
}

void NetworkGraph::setInputsLabels(sharedConstTensors &inputs,
                                   sharedConstTensors &labels) {
  /**
   * @note This code written based on the assumption that he graph consists
   * with only one subgraph requiring inputs and one subgraph requiring labels.
   * This should be updated later.
   */
  (*cbegin())->setInputs(inputs);
  (*crbegin())->setLabels(labels);
}

std::vector<Tensor> NetworkGraph::getOutputTensors() const {
  /**
   * @note This code written based on the assumption that he graph consists
   * with only one default subgraph node. It needs to be updated.
   */
  return (*crbegin())->getOutputTensors();
}

void NetworkGraph::flushCache() { tensor_manager->flushCache(); }

void NetworkGraph::flushCacheExcept(unsigned int order) {
  tensor_manager->flushCacheExcept(order);
}

void NetworkGraph::LoadTensors(unsigned int order) {
  tensor_manager->LoadTensors(order);
}

bool NetworkGraph::checkLoadComplete(unsigned int order) {
  return tensor_manager->checkLoadComplete(order);
}

bool NetworkGraph::checkUnloadComplete(unsigned int order) {
  return tensor_manager->checkUnloadComplete(order);
}

void NetworkGraph::UnloadTensors(unsigned int order) {
  tensor_manager->UnloadTensors(order);
}

void NetworkGraph::requestOptimizerVariable(
  std::function<std::vector<TensorDim>(const TensorDim &)> cb,
  bool request_only_trainable) {
  for (auto it = cbegin(); it != cend(); ++it)
    (*it)->requestOptimizerVariable(cb, request_only_trainable);
}

void NetworkGraph::resetLossScale(float scale) {
  for (auto it = cbegin(); it != cend(); ++it)
    (*it)->resetLossScale(scale);
}

} /* namespace nntrainer */
