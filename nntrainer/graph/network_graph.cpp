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

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

int NetworkGraph::compile(const std::string &loss_type) {
  return graph.compile(loss_type);
}

void NetworkGraph::setBatchSize(unsigned int batch_size) {
  graph.setBatchSize(batch_size);
}

void NetworkGraph::applyGradients(
  LayerNode *node, const std::function<void(Weight &)> &apply_func) {
  SubGraphBase::applyGradients(node, apply_func);
}

sharedConstTensors NetworkGraph::forwarding(
  bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  return graph.forwarding(training, forwarding_op, stop_cb, userdata);
}

sharedConstTensors NetworkGraph::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  return graph.incremental_forwarding(from, to, training, forwarding_op,
                                      stop_cb, userdata);
}

bool NetworkGraph::backwarding(
  int iteration,
  std::function<void(std::shared_ptr<LayerNode>, bool)> &forwarding_op,
  std::function<bool(std::shared_ptr<LayerNode>, int)> &backwarding_op,
  std::function<void(Weight &, int)> &lazy_apply_grad_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  return graph.backwarding(iteration, forwarding_op, backwarding_op,
                           lazy_apply_grad_op, stop_cb, userdata);
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void NetworkGraph::allocateTensors(ExecutionMode exec_mode_) {
  graph.allocateTensors(exec_mode_);
}

std::vector<TensorDim> NetworkGraph::getInputDimension() const {
  return graph.getInputDimension();
}

unsigned int NetworkGraph::getBatchSize() const { return graph.getBatchSize(); }

std::vector<TensorDim> NetworkGraph::getOutputDimension() const {
  return graph.getOutputDimension();
}

std::vector<std::shared_ptr<LayerNode>>
NetworkGraph::getUnsortedLayers(const std::string &input_layer,
                                const std::string &output_layer) const {
  return graph.getUnsortedLayers(input_layer, output_layer);
}

std::vector<std::shared_ptr<LayerNode>> NetworkGraph::getLayerNodes() const {
  return graph.getLayerNodes();
}

void NetworkGraph::addLayer(std::shared_ptr<LayerNode> layer) {
  graph.addLayer(layer);
}

std::vector<Var_Grad *>
NetworkGraph::finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                              const std::vector<Var_Grad *> &prev_inputs) {
  return graph.finalizeContext(lnode, prev_inputs);
}

std::vector<Var_Grad *>
NetworkGraph::refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                                const std::vector<Var_Grad *> &prev_inputs) {
  return graph.refinalizeContext(lnode, prev_inputs);
}

#ifdef ENABLE_TEST

std::map<std::string, std::vector<unsigned int>>
NetworkGraph::getLayerExecutionOrders(const std::shared_ptr<LayerNode> &lnode) {
  return graph.getLayerExecutionOrders(lnode);
}

#endif // ENABLE_TEST

int NetworkGraph::initialize(ExecutionMode mode,
                             const std::vector<Connection> &model_input_names,
                             const std::vector<Connection> &model_label_names) {

  return graph.initialize(mode, model_input_names, model_label_names);
}

int NetworkGraph::reinitialize(
  const std::vector<Connection> &model_input_names,
  const std::vector<Connection> &model_label_names) {
  return graph.reinitialize(model_input_names, model_label_names);
}

void NetworkGraph::setInputsLabels(const std::vector<Tensor> &inputs,
                                   const std::vector<Tensor> &labels) {
  graph.setInputsLabels(inputs, labels);
}

void NetworkGraph::setInputsLabels(sharedConstTensors &inputs,
                                   sharedConstTensors &labels) {
  graph.setInputsLabels(inputs, labels);
}

std::vector<Tensor> NetworkGraph::getOutputTensors() const {
  return graph.getOutputTensors();
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
  graph.requestOptimizerVariable(cb, request_only_trainable);
}

void NetworkGraph::resetLossScale(float scale) { graph.resetLossScale(scale); }

} /* namespace nntrainer */
