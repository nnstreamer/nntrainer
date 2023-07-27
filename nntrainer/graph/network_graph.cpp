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

#include "graph_node.h"
#include "tensor.h"
#include <cmath>
#include <stdexcept>
#include <string>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <connection.h>
#include <cross_entropy_loss_layer.h>
#include <cross_entropy_sigmoid_loss_layer.h>
#include <cross_entropy_softmax_loss_layer.h>
#include <flatten_layer.h>
#include <grucell.h>
#include <identity_layer.h>
#include <input_layer.h>
#include <layer_node.h>
#include <layer_normalization_layer.h>
#include <lstmcell.h>
#include <multiout_layer.h>
#include <network_graph.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>
#include <rnn.h>
#include <rnncell.h>
#include <split_layer.h>
#include <time_dist.h>
#include <tracer.h>
#include <util_func.h>

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

int NetworkGraph::compile(const std::string &loss_type) {
  int status = ML_ERROR_NONE;

  status = isCompilable();
  NN_RETURN_STATUS();

  try {
    setOutputConnections();
  } catch (std::exception &e) {
    ml_loge("setting output layer failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  graph.realizeInputOutputNode();

  try {
    /// @todo realize loss beforehand
    status = addLossLayer(loss_type);
    NN_RETURN_STATUS();
  } catch (const std::exception &e) {
    ml_loge("%s", e.what());
    status = ML_ERROR_INVALID_PARAMETER;
    NN_RETURN_STATUS();
  }

  graph.topologicalSort();

  setExecutionOrder();
  forward_iter_end = (*(cend() - 1)).get();

  inPlaceOptimize();

  status = checkCompiledGraph();
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

void NetworkGraph::setExecutionOrder() {
  auto backward_order = graph.size();
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

void NetworkGraph::addLayerNode(std::unique_ptr<Layer> layer) {
  graph.addNode(std::make_unique<LayerNode>(std::move(layer)));
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
        LNODE(graph.getNode(output_layer_node->getInputConnectionName(0)));
    }

    std::shared_ptr<LayerNode> lnode = createLayerNode(loss_type);
    graph.ensureName(*lnode);

    if (second_to_last_layer_node->getDistribute()) {
      lnode->setProperty({"distribute=true"});
    }

    /// @todo remove this by add loss at realization
    second_to_last_layer_node->setOutputLayers({lnode->getName()});
    lnode->setProperty(
      {"input_layers=" + second_to_last_layer_node->getName()});

    if (is_cross_entropy_loss) {
      graph.replaceNode(output_layer_node, lnode);
    } else {
      graph.addNode(lnode, false);
    }
    graph.replaceOutputNode(i, lnode);
  }

  return ML_ERROR_NONE;
}

void NetworkGraph::setOutputConnections() {
  for (auto layer_iter = cbegin(); layer_iter != cend(); layer_iter++) {
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
    auto ln = LNODE(graph.getNode(node_name)).get();
    ln->needsCalcDerivative(true);
  }
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
    if ((*iter)->isFinalized()) {
      /// resize tensors spec
      /// @todo remove below, if custom tensor needs to change dimension
      /// according to the tensor, it must be done explicitly, or at least have
      /// a property to control the behavior
      const RunLayerContext &context = (*iter)->getRunContext();
      for (unsigned int idx = 0; idx < context.getNumTensors(); idx++) {
        auto const &ts = context.getTensor(idx);
        tensor_manager->setBatchSize(ts.getName(), ts.getDim().batch());
        if (context.tensorHasGradient(idx)) {
          auto const &ts_grad = context.getTensorGrad(idx);
          tensor_manager->setBatchSize(ts_grad.getName(),
                                       ts_grad.getDim().batch());
        }
      }
      /// override setting batch as per request
      (*iter)->setBatch(batch_size);
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

void NetworkGraph::applyGradients(
  LayerNode *node, const std::function<void(Weight &)> &apply_func) {

  if (!node->getTrainable())
    return;

  TRACE_MEMORY() << node->getName() + ": AG";
  TRACE_TIME() << node->getName() + ": AG";

  auto &rc = node->getRunContext();
  auto num_weight = rc.getNumWeights();
  for (unsigned i = 0; i < num_weight; ++i) {
    if (!rc.weightHasGradient(i)) {
      continue;
    }

    if (!rc.isGradientLastAccess(i)) {
      /// @note instead of checking the last access of the weight, checking
      /// if weights are dependent to others to minimize overhead.
      /// this logic assume that the source of the dependent weight must be
      /// prior to the dependent.
      continue;
    }

    if (rc.isGradientClipByGlobalNorm(i)) {
      /**
       * @note the weights whose gradient are to be clipped by global norm will
       * be clipped at once at the end of iteration and applied then.
       */
      continue;
    }

    apply_func(rc.getWeightObject(i));
  }
}

sharedConstTensors NetworkGraph::forwarding(
  bool training,
  std::function<void(std::shared_ptr<LayerNode>, bool)> forwarding_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  for (auto iter = cbegin(); iter != cend() && !stop_cb(userdata); iter++) {
    auto &ln = *iter;
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    forwarding_op(*iter, training);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));
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
  std::function<void(std::shared_ptr<LayerNode>, int)> &backwarding_op,
  std::function<void(Weight &, int)> &apply_grad_clip_op,
  std::function<bool(void *userdata)> stop_cb, void *userdata) const {
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

  for (auto iter = iter_begin; iter != iter_end && !stop_cb(userdata); iter++) {
    auto &ln = *iter;
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    backwarding_op(ln, iteration);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));
  }

  /** perform clipping of the gradients by global norm if any */
  if (clip_weights.empty())
    return;

  /** calculate the global norm */
  Tensor global_norm_t(
    TensorDim({1u, 1u, 1u, (unsigned int)clip_weights.size()}));
  float *global_norm_data = global_norm_t.getData();
  for (unsigned int idx = 0; idx < clip_weights.size(); idx++) {
    auto const &w = clip_weights[idx];
    global_norm_data[idx] = w->getGradientNorm();
  }
  float global_norm = global_norm_t.l2norm();
  /** apply the gradient with the above global norm */
  for (auto w : clip_weights) {
    w->clipGradientByGlobalNorm(global_norm);
  }
  /** apply the gradient with the above global norm */
  for (auto w : clip_weights) {
    apply_grad_clip_op(*w, iteration);
  }
}

LayerNode *NetworkGraph::computeBackwardEnd() {
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
    tensor_manager->allocateTensors(
      std::get<3>(backward_iter_end->getExecutionOrder()));
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
    return lnode->getType() == FlattenLayer::type ||
           lnode->getType() == IdentityLayer::type;
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
      return (lnode->getType() == BatchNormalizationLayer::type) ||
             (lnode->getType() == LayerNormalizationLayer::type);
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
    for (auto i = 0u, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      const auto &input_name = lnode->getInputConnectionName(i);
      if (getLayerNode(input_name)->executeInPlace() == InPlace::RESTRICTING)
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
   * This is a generic case where the layer can support in-place but will
   * modify its input in-place. This includes layers like activation, etc.
   * Apply checks below to ensure that the layers can work in-place:
   * - if any of the input layer are restriction, then this layer cannot work
   *   as layers behind this layer have added restrictions.
   * - if all of the input layers are either not inplace or have no
   * restrictions, then this layer can operate in-place.
   *
   * @note Conditions to decide the type of inplace for this layer:
   * This is a generic case, and always restrictions on the next nodes to be
   * not inplace.
   *
   * @note This logic is prone to change as more layers are allowed to
   * work in-place such as concat layer, split layer, addition layer, dropout
   * layer, etc.
   *
   * @todo This logic sets layers to in-place one-by-one as they arrive. However
   * setting some layers to in-place can save more memory than others (like
   * multiout layer vs activation layer). The layers need to sorted based on the
   * memory save they provide and then make them in-place in that order.
   */
  if (lnode->getType() == ActivationLayer::type ||
      lnode->getType() == BatchNormalizationLayer::type ||
      lnode->getType() == LayerNormalizationLayer::type) {
    for (auto i = 0u, num_node = lnode->getNumInputConnections(); i < num_node;
         ++i) {
      if (getLayerNode(lnode->getInputConnectionName(i))->executeInPlace() ==
          InPlace::RESTRICTING)
        return InPlace::NONE;
    }

    /**
     * if the layer does io_independent_backwarding where the input and output
     * is not required during backwarding, then it is a non-restricting in-place
     * layer.
     */
    if (io_independent_backwarding(lnode))
      return InPlace::NON_RESTRICTING;

    return InPlace::RESTRICTING;
  }

  /**
   * if the layer's input and output type is not FP32, then it cannot be
   * inplace. We assume that the input is always FP32.
   */
  if (lnode->getInputConnections().empty()) {
    if (!istrequal(getTensorType()[2], "FP32"))
      return InPlace::NONE;
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
   * @todo for layers which support in-place, both variables and gradients
   * will be shared.
   *
   * @todo add a check here is the layer being checked here can support
   * in-place or not
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
  auto init_context = lnode->finalize(input_dims, getTensorType());

  /**
   * Request manager for either a pre-allocated output as input or a newly
   * allocated output. This is necessary for manager to know when this output
   * node is going to be used.
   */
  std::vector<std::string> input_names;
  input_names.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_names),
                 [](auto const &vg) { return vg->getName(); });
  const std::vector<Var_Grad *> &inputs = tensor_manager->requestInputs(
    gnode, init_context.getInputDimensions(), input_names);

  /** In-Place optimizations */
  /**
   * Request manager for either a pre-allocated input as output or a newly
   * allocated output. This is necessary for manager to know when this output
   * node is going to be used with in-place optimizations.
   */
  auto out_specs = init_context.getOutSpecs();
  /// @note try move inplace control to finalize
  bool shared_var = false, shared_grad = false;
  if (lnode->executeInPlace() != InPlace::NONE) {
    setInplaceSharedMemoryConfigByLayer(lnode, shared_var, shared_grad);
    for (unsigned int i = 0; i < out_specs.size(); ++i) {
      auto &s = out_specs.at(i);
      if (shared_var) {
        s.variable_spec.request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.variable_spec.reference_name = inputs[i]->getName();
        } else {
          s.variable_spec.reference_name = inputs[0]->getName();
        }
      }
      if (shared_grad && s.gradient_spec) {
        s.gradient_spec->request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.gradient_spec->reference_name = inputs[i]->getGradientName();
        } else {
          s.gradient_spec->reference_name = inputs[0]->getGradientName();
        }
      }
    }
  }
  if (lnode->requireLabel()) {
    NNTR_THROW_IF(out_specs.size() != 1, std::invalid_argument)
      << "out specification size must be 1 for label layer for now, "
      << lnode->getName() << " out spec size: " << out_specs.size();
    NNTR_THROW_IF(out_specs[0].gradient_spec == nullptr, std::invalid_argument)
      << "label space does not exist for " << lnode->getName();
    out_specs[0].gradient_spec->request_type =
      TensorSpecV2::RequestType::PLACEHOLDER;
  }

  /// @note below needs to be enabled only for inference mode, but need decision
  /// if we are going to separate inference initialization from train
  /// initialization this might not worth optimize because in general output of
  /// a neuralnet is very small
  if (lnode->getOutputConnections().size() == 0u) {
    std::for_each(out_specs.begin(), out_specs.end(),
                  [this](VarGradSpecV2 &spec) {
                    spec.variable_spec.additional_exec_order.push_back(
                      std::get<0>(forward_iter_end->getExecutionOrder()));
                  });
  }

  if (lnode->getType() == RNNCellLayer::type or
      lnode->getType() == LSTMCellLayer::type or
      lnode->getType() == GRUCellLayer::type) {
    std::for_each(
      out_specs.begin(), out_specs.end(), [this](VarGradSpecV2 &spec) {
        spec.variable_spec.ls = TensorLifespan::FORWARD_GRAD_LIFESPAN;
      });
  }

  const std::vector<Var_Grad *> &outputs = tensor_manager->requestTensors(
    out_specs, Manager::TensorGroupType::OUTPUT, lnode->getExecutionOrder(),
    lnode->getName());

  /** create shared weight names if requested */
  std::vector<std::string> shared_weight_names;
  std::vector<std::string> shared_tensor_names;
  if (auto shared_node_str = lnode->getSharedFrom(); !shared_node_str.empty()) {
    /// @note below is commented but kept from quick fix to be referenced for
    /// later(#1707)
    // auto shared_node = getLayerNode(shared_node_str).get();
    // NNTR_THROW_IF(shared_node == nullptr, std::invalid_argument)
    //   << "shared_node requested but it is not registered in the graph,
    //   name:
    //   "
    //   << shared_node_str << " requested from " << lnode->getName();
    // NNTR_THROW_IF(shared_node->getType() != lnode->getType(),
    //               std::invalid_argument)
    //   << " shared_node and lnode type mismatch, source node type: "
    //   << shared_node->getType() << " depedent node type: " <<
    //   lnode->getType()
    //   << " depedent node name: " << lnode->getName();
    // NNTR_THROW_IF(!shared_node->isFinalized(), std::invalid_argument)
    //   << "shared node must be prior to the dependent node and it should be
    //   "
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
      shared_weight_names.emplace_back(std::get<7>(w_specs.at(i)));
    }
  }

  lnode->configureRunContext(
    // TODO: update weights spec for trainable based on layer trainable prop
    tensor_manager->requestWeights(gnode, init_context.getWeightsSpec(),
                                   lnode->getTrainable(), shared_weight_names),
    inputs, outputs,
    tensor_manager->requestTensors(gnode, init_context.getTensorsSpec(),
                                   lnode->getTrainable(), shared_tensor_names));

  return outputs;
}

std::vector<Var_Grad *>
NetworkGraph::refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                                const std::vector<Var_Grad *> &prev_inputs) {
  const GraphNode &gnode = *lnode.get();
  std::vector<TensorDim> input_dims;
  input_dims.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_dims),
                 [](const Var_Grad *vg) { return vg->getDim(); });

  /** refinalize the layer and get the final context */
  auto init_context = lnode->refinalize(input_dims);

  /**
   * Request manager for either a pre-allocated output as input or a newly
   * allocated output. This is necessary for manager to know when this output
   * node is going to be used.
   */
  std::vector<std::string> input_names;
  input_names.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_names),
                 [](auto const &vg) { return vg->getName(); });
  const std::vector<Var_Grad *> &inputs = tensor_manager->requestInputs(
    gnode, init_context.getInputDimensions(), input_names);

  /** In-Place optimizations */
  /**
   * Request manager for either a pre-allocated input as output or a newly
   * allocated output. This is necessary for manager to know when this output
   * node is going to be used with in-place optimizations.
   */
  auto out_specs = init_context.getOutSpecs();
  /// @note try move inplace control to finalize
  bool shared_var = false, shared_grad = false;
  if (lnode->executeInPlace() != InPlace::NONE) {
    setInplaceSharedMemoryConfigByLayer(lnode, shared_var, shared_grad);
    for (unsigned int i = 0; i < out_specs.size(); ++i) {
      auto &s = out_specs.at(i);
      if (shared_var) {
        s.variable_spec.request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.variable_spec.reference_name = inputs[i]->getName();
        } else {
          s.variable_spec.reference_name = inputs[0]->getName();
        }
      }
      if (shared_grad && s.gradient_spec) {
        s.gradient_spec->request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.gradient_spec->reference_name = inputs[i]->getGradientName();
        } else {
          s.gradient_spec->reference_name = inputs[0]->getGradientName();
        }
      }
    }
  }
  if (lnode->requireLabel()) {
    NNTR_THROW_IF(out_specs.size() != 1, std::invalid_argument)
      << "out specification size must be 1 for label layer for now, "
      << lnode->getName() << " out spec size: " << out_specs.size();
    NNTR_THROW_IF(out_specs[0].gradient_spec == nullptr, std::invalid_argument)
      << "label space does not exist for " << lnode->getName();
    out_specs[0].gradient_spec->request_type =
      TensorSpecV2::RequestType::PLACEHOLDER;
  }

  /// @note below needs to be enabled only for inference mode, but need decision
  /// if we are going to separate inference initialization from train
  /// initialization this might not worth optimize because in general output of
  /// a neuralnet is very small
  if (lnode->getOutputConnections().size() == 0u) {
    std::for_each(out_specs.begin(), out_specs.end(),
                  [this](VarGradSpecV2 &spec) {
                    spec.variable_spec.additional_exec_order.push_back(
                      std::get<0>(forward_iter_end->getExecutionOrder()));
                  });
  }

  if (lnode->getType() == RNNCellLayer::type or
      lnode->getType() == LSTMCellLayer::type or
      lnode->getType() == GRUCellLayer::type) {
    std::for_each(
      out_specs.begin(), out_specs.end(), [this](VarGradSpecV2 &spec) {
        spec.variable_spec.ls = TensorLifespan::FORWARD_GRAD_LIFESPAN;
      });
  }

  const std::vector<Var_Grad *> &outputs = tensor_manager->requestTensors(
    out_specs, Manager::TensorGroupType::OUTPUT, lnode->getExecutionOrder(),
    lnode->getName());

  /** create shared weight names if requested */
  std::vector<std::string> shared_weight_names;
  std::vector<std::string> shared_tensor_names;
  if (auto shared_node_str = lnode->getSharedFrom(); !shared_node_str.empty()) {
    /// @note below is commented but kept from quick fix to be referenced for
    /// later(#1707)
    // auto shared_node = getLayerNode(shared_node_str).get();
    // NNTR_THROW_IF(shared_node == nullptr, std::invalid_argument)
    //   << "shared_node requested but it is not registered in the graph,
    //   name:
    //   "
    //   << shared_node_str << " requested from " << lnode->getName();
    // NNTR_THROW_IF(shared_node->getType() != lnode->getType(),
    //               std::invalid_argument)
    //   << " shared_node and lnode type mismatch, source node type: "
    //   << shared_node->getType() << " depedent node type: " <<
    //   lnode->getType()
    //   << " depedent node name: " << lnode->getName();
    // NNTR_THROW_IF(!shared_node->isFinalized(), std::invalid_argument)
    //   << "shared node must be prior to the dependent node and it should be
    //   "
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
      shared_weight_names.emplace_back(std::get<7>(w_specs.at(i)));
    }
  }

  auto weights = lnode->getRunContext().getWeights();
  lnode->configureRunContext(
    // TODO: update weights spec for trainable based on layer trainable prop
    weights, inputs, outputs,
    tensor_manager->requestTensors(gnode, init_context.getTensorsSpec(),
                                   lnode->getTrainable(), shared_tensor_names));

  return outputs;
}

#ifdef ENABLE_TEST

std::map<std::string, std::vector<unsigned int>>
NetworkGraph::getLayerExecutionOrders(const std::shared_ptr<LayerNode> &lnode) {
  auto init_context = lnode->getInitContext();
  auto out_specs = init_context.getOutSpecs();
  auto weight_specs = init_context.getWeightsSpec();
  auto tensor_specs = init_context.getTensorsSpec();

  std::map<std::string, std::vector<unsigned int>> exec_orders;

  for (auto &spec : out_specs) {
    const auto name = lnode->getName() + ":" + spec.variable_spec.name;
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
    const auto name = std::get<const std::string>(spec);
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
    const auto name = std::get<const std::string>(spec);
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

int NetworkGraph::initialize(const std::vector<Connection> &model_input_names,
                             const std::vector<Connection> &model_label_names) {

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

    if (profile_keys.find(lnode->getType()) == profile_keys.end()) {
      int event_key = 0;
      PROFILE_TIME_REGISTER_EVENT(event_key, lnode->getType());
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

    for (auto i = 0u, num_node = lnode->getNumOutputConnections(); i < num_node;
         ++i) {
      auto conn = lnode->getOutputConnection(i);
      if (!conn) {
        ml_logi("out connection not defined for  %s, %u",
                lnode->getName().c_str(), i);
        continue;
      }

      auto sink_node = getLayerNode(conn->getName());
      [[maybe_unused]] auto [it, b] =
        input_map.try_emplace({sink_node->getName(), {}});

      NNTR_THROW_IF(sink_node->getInputConnectionName(conn->getIndex()) !=
                      lnode->getName(),
                    std::invalid_argument)
        << "node pair does not match between " << lnode->getName() << ' '
        << sink_node->getName();

      auto &sink_tensors = it->second;
      sink_tensors.resize(sink_node->getNumInputConnections());
      sink_tensors[conn->getIndex()] = outputs[i];
    }
  }

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    auto const &lnode = getSortedLayerNode(idx);
    auto &rc = lnode->getRunContext();
    auto first_grad_access = std::get<1>(lnode->getExecutionOrder());
    auto last_grad_access = std::get<3>(lnode->getExecutionOrder());
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
        /**
         * if the gradient is to be clipped by global norm, then the last access
         * is by clipping itself. However, as clipping is not a layer and does
         * not contain any weights, such weights never get assigned
         * gradient_last_access. This is a quick hotfix.
         * TODO: make an independent clipping layer which will execute at the
         * end, and will share ownership of weights which it will clip. This
         * will remove this hot fix, and also remove the checks of if weights
         * require clipping.
         */
        if (tensor_manager->isLastAccess(rc.getWeightGrad(i).getName(),
                                         last_grad_access) ||
            (rc.isGradientClipByGlobalNorm(i) &&
             tensor_manager->isSecondLastAccess(rc.getWeightGrad(i).getName(),
                                                last_grad_access))) {
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

  auto identify_external_tensors = [this](const std::vector<Connection> &conns,
                                          auto &&pred, auto &&identify) {
    if (conns.empty()) {
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        /// when name is empty, we identify everything as the node, all of
        /// them must be having identical dimensions
        identify(lnode);
      }
    } else {
      for (auto &conn : conns) {
        auto lnode = getLayerNode(conn.getName()).get();
        NNTR_THROW_IF(!pred(lnode), std::invalid_argument)
          << "given node is not of that kind, name: " << conn.getName();
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
      NNTR_THROW_IF(num_node_of_kind != conns.size(), std::invalid_argument)
        << "conns given but there are not identified node of the kind, num "
           "node of kind: "
        << num_node_of_kind << " identifier size: " << conns.size();
    }
  };

  identify_external_tensors(model_input_names, is_input_node,
                            identify_as_model_input);
  identify_external_tensors(model_label_names, is_label_node,
                            identify_as_model_label);

  /** mark the nodes which will be backwarded during the graph operation */
  try {
    markNodesForBackwarding();
    backward_iter_end = computeBackwardEnd();
  } catch (std::exception &e) {
    ml_loge(
      "Backwarding required from layer which doesn't support backwarding: %s",
      e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** select weights which would require clipping of the gradients by global
   * norm if any */
  clip_weights = tensor_manager->getWeights([](const Weight *w) {
    return w->hasGradient() && w->isGradientLastAccess() &&
           w->isGradientClipByGlobalNorm();
  });

  return ML_ERROR_NONE;
}

int NetworkGraph::reinitialize(
  const std::vector<Connection> &model_input_names,
  const std::vector<Connection> &model_label_names) {
  input_dims.clear();
  label_dims.clear();
  tensor_manager->reinitialize();

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

    if (profile_keys.find(lnode->getType()) == profile_keys.end()) {
      int event_key = 0;
      PROFILE_TIME_REGISTER_EVENT(event_key, lnode->getType());
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
     * Reinitialize all the layers, allocate output tensors for each layer
     * init2and add optimizer related weights for the layer
     */
    const std::vector<Var_Grad *> &outputs = refinalizeContext(lnode, inputs);

    /** no need to update input_map for the last layer */
    if (idx == graph.size() - 1)
      break;

    for (auto i = 0u, num_node = lnode->getNumOutputConnections(); i < num_node;
         ++i) {
      auto conn = lnode->getOutputConnection(i);
      if (!conn) {
        ml_logi("out connection not defined for  %s, %u",
                lnode->getName().c_str(), i);
        continue;
      }

      auto sink_node = getLayerNode(conn->getName());
      [[maybe_unused]] auto [it, b] =
        input_map.try_emplace({sink_node->getName(), {}});

      NNTR_THROW_IF(sink_node->getInputConnectionName(conn->getIndex()) !=
                      lnode->getName(),
                    std::invalid_argument)
        << "node pair does not match between " << lnode->getName() << ' '
        << sink_node->getName();

      auto &sink_tensors = it->second;
      sink_tensors.resize(sink_node->getNumInputConnections());
      sink_tensors[conn->getIndex()] = outputs[i];
    }
  }

  for (unsigned int idx = 0; idx < graph.size(); ++idx) {
    auto const &lnode = getSortedLayerNode(idx);
    auto &rc = lnode->getRunContext();
    auto first_grad_access = std::get<1>(lnode->getExecutionOrder());
    auto last_grad_access = std::get<3>(lnode->getExecutionOrder());
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
        /**
         * if the gradient is to be clipped by global norm, then the last access
         * is by clipping itself. However, as clipping is not a layer and does
         * not contain any weights, such weights never get assigned
         * gradient_last_access. This is a quick hotfix.
         * TODO: make an independent clipping layer which will execute at the
         * end, and will share ownership of weights which it will clip. This
         * will remove this hot fix, and also remove the checks of if weights
         * require clipping.
         */
        if (tensor_manager->isLastAccess(rc.getWeightGrad(i).getName(),
                                         last_grad_access) ||
            (rc.isGradientClipByGlobalNorm(i) &&
             tensor_manager->isSecondLastAccess(rc.getWeightGrad(i).getName(),
                                                last_grad_access))) {
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

    // input_list.push_back(node->getInput(0).getName());
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
    // output_list.push_back(node->getOutput(0).getName());
    // label_list.push_back(node->getOutputGrad(0).getName());
    label_dims.push_back(node->getOutputDimensions()[0]);
  };

  auto identify_external_tensors = [this](const std::vector<Connection> &conns,
                                          auto &&pred, auto &&identify) {
    if (conns.empty()) {
      for (unsigned int i = 0; i < graph.size(); ++i) {
        auto lnode = getSortedLayerNode(i).get();
        if (!pred(lnode)) {
          continue;
        }
        /// when name is empty, we identify everything as the node, all of
        /// them must be having identical dimensions
        identify(lnode);
      }
    } else {
      for (auto &conn : conns) {
        auto lnode = getLayerNode(conn.getName()).get();
        NNTR_THROW_IF(!pred(lnode), std::invalid_argument)
          << "given node is not of that kind, name: " << conn.getName();
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
      NNTR_THROW_IF(num_node_of_kind != conns.size(), std::invalid_argument)
        << "conns given but there are not identified node of the kind, num "
           "node of kind: "
        << num_node_of_kind << " identifier size: " << conns.size();
    }
  };

  identify_external_tensors(model_input_names, is_input_node,
                            identify_as_model_input);
  identify_external_tensors(model_label_names, is_label_node,
                            identify_as_model_label);

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

void NetworkGraph::flushCache() { tensor_manager->flushCache(); }

void NetworkGraph::flushCacheExcept(unsigned int order) {
  tensor_manager->flushCacheExcept(order);
}

void NetworkGraph::requestOptimizerVariable(
  std::function<std::vector<TensorDim>(const TensorDim &)> cb,
  bool request_only_trainable) {
  for (auto const &w : tensor_manager->getWeights()) {
    if (w->isGradientLastAccess() && w->hasGradient()) {
      const TensorDim &dim = w->getDim();
      std::vector<TensorDim> dims = cb(dim);
      w->setOptimizerVariables(tensor_manager->requestWeightOptimizerVariables(
        dims, w->getName(), TensorLifespan::MAX_LIFESPAN,
        w->isGradientClipByGlobalNorm(), Tensor::Initializer::ZEROS));
    }
  }
}

} /* namespace nntrainer */
