// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    subgraph_cpu.cpp
 * @date    07 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is Network SubGraphCpu Class for Neural Network
 *
 * @todo    Support multi-input graph.
 */

#include <cmath>
#include <stdexcept>
#include <string>

#include <flatten_layer.h>
#include <grucell.h>
#include <identity_layer.h>
#include <layer_normalization_layer.h>
#include <lstmcell.h>
#include <multiout_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_context.h>
#include <profiler.h>
#include <rnn.h>
#include <rnncell.h>
#include <time_dist.h>
#include <tracer.h>
#include <util_func.h>

#include <layer_node.h>
#include <subgraph_cpu.h>

#define LNODE(x) std::static_pointer_cast<LayerNode>(x)

namespace nntrainer {

int SubGraphCpu::compile(const std::string &loss_type) {
  int status = ML_ERROR_NONE;

  status = isCompilable();
  NN_RETURN_STATUS();

  try {
    setOutputConnections();
  } catch (std::exception &e) {
    ml_loge("setting output layer failed, reason: %s", e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  subgraph.realizeInputOutputNode();

  if (exec_mode != ExecutionMode::INFERENCE) {
    try {
      /// @todo realize loss beforehand
      status = addLossLayer(loss_type);
      NN_RETURN_STATUS();
    } catch (const std::exception &e) {
      ml_loge("%s", e.what());
      status = ML_ERROR_INVALID_PARAMETER;
      NN_RETURN_STATUS();
    }
  } else {
    if (!loss_type.empty()) {
      ml_loge(
        "Warning : Loss type is given in inference mode. Ignoring loss type.");
    }
  }

  subgraph.topologicalSort();

  setExecutionOrder();
  forward_iter_end = (*(cend() - 1)).get();

  inPlaceOptimize();

  status = checkCompiledGraph();
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

void SubGraphCpu::setBatchSize(unsigned int batch_size) {
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

void SubGraphCpu::applyGradients(LayerNode *node, int iteration,
                                 std::shared_ptr<OptimizerWrapped> opt) {

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

    if (rc.isGradientClipByGlobalNorm(i) || rc.isMixedPrecision(i)) {
      /**
       * @note the weights whose gradient are to be clipped by global norm will
       * be clipped at once at the end of iteration and applied then.
       * For those weights where mixed precision is uesed, their gradient
       * updates might be delayed until they confirm whether their loss scales
       * are appropeiate.
       */
      continue;
    }

    auto &w = rc.getWeightObject(i);
    w.calcRegularizationGradient();
    w.calcWeightDecayGradient();
    RunOptimizerContext opt_context(&w, iteration,
                                    opt->getLearningRate(iteration));
    opt->applyGradient(opt_context);
  }
}

sharedConstTensors
SubGraphCpu::forwarding(bool training,
                        std::function<bool(void *userdata)> stop_cb,
                        void *userdata, bool swap_mode) {
  for (auto iter = cbegin(); iter != cend() && !stop_cb(userdata); iter++) {
    auto &ln = *iter;
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    forwarding_op(*iter, training, swap_mode);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));
  }

  sharedConstTensors out;
  for (unsigned int i = 0; i < subgraph.getNumOutputNodes(); ++i) {
    auto const &output_layer_node = LNODE(subgraph.getOutputNode(i));
    for (unsigned int j = 0; j < output_layer_node->getNumOutputs(); ++j) {
      out.push_back(MAKE_SHARED_TENSOR(output_layer_node->getOutput(j)));
    }
  }

  return out;
}

sharedConstTensors SubGraphCpu::incremental_forwarding(
  unsigned int from, unsigned int to, bool training,
  std::function<bool(void *userdata)> stop_cb, void *userdata) {
  for (auto iter = cbegin(); iter != cend() && !stop_cb(userdata); iter++) {
    auto &ln = *iter;
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    incremental_forwarding_op(*iter, from, to, training);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));
  }

  sharedConstTensors out;
  for (unsigned int i = 0; i < subgraph.getNumOutputNodes(); ++i) {
    auto const &output_layer_node = LNODE(subgraph.getOutputNode(i));
    for (unsigned int j = 0; j < output_layer_node->getNumOutputs(); ++j) {
      out.push_back(MAKE_SHARED_TENSOR(output_layer_node->getOutput(j)));
    }
  }

  return out;
}

bool SubGraphCpu::backwarding(int iteration,
                              std::function<bool(void *userdata)> stop_cb,
                              void *user_data, bool is_grad_opt_mode,
                              std::shared_ptr<OptimizerWrapped> opt) {
  /**
   * last layer backwarding is run out of this loop
   */
  auto iter_begin = getBackwardingBeginIter();
  auto iter_end = getBackwardingEndIter();
  bool is_valid = true;

  /// there is no layer to train, so backwarding is essentially noop
  if (iter_begin == iter_end) {
    return true;
  }

  auto const &lptr_begin = (*iter_begin);
  // graph_const_reverse_iterator
  auto iter_ = iter_begin;

  if (lptr_begin->requireLabel() == false)
    throw std::runtime_error(
      "Error: last layer does not accept label, we can't train");

  for (iter_ = iter_begin; iter_ != iter_end && !stop_cb(user_data); iter_++) {
    auto &ln = *iter_;
    PROFILE_TIME_START(profile_keys.at(ln->getType()));
    is_valid =
      backwarding_op(ln, iteration, stop_cb, user_data, is_grad_opt_mode, opt);
    PROFILE_TIME_END(profile_keys.at(ln->getType()));

    if (!is_valid) {
      break;
    }
  }

  if (!is_valid) {
    /** if has NaN
     * 1. reset the loss scale. : @todo Backoff_factor : default --> 0.5
     * 2. run forwarding from cur_iter to cend() && !stop_cb(userdata);
     * 3. return false --> run backwarding again;
     */
    float scale = (*iter_)->getRunContext().getLossScale();

    NNTR_THROW_IF(scale - 1.0f < 10e-6, std::invalid_argument)
      << "Loss Scale Factor is 1.0f";

    float s = scale > 1.5f ? scale * 0.5f : 1.0f;

    resetLossScale(s);

    auto f_iter = cbegin() + subgraph.getSortedNodeIdx((*iter_)->getName());

    for (auto iter = f_iter; iter != cend() && !stop_cb(user_data); iter++) {
      auto &ln = *iter;
      ln->reStoreData(true);
    }

    for (auto iter = f_iter; iter != cend() && !stop_cb(user_data); iter++) {
      auto &ln = *iter;
      PROFILE_TIME_START(profile_keys.at(ln->getType()));
      forwarding_op(*iter, true);
      PROFILE_TIME_END(profile_keys.at(ln->getType()));
    }

    return false;
  }

  /** perform clipping of the gradients by global norm if any */
  if (lazy_weights.empty())
    return true;

  if (is_clip_grad) {
    /** calculate the global norm */
    Tensor global_norm_t(
      TensorDim({1u, 1u, 1u, (unsigned int)lazy_weights.size()}));
    float *global_norm_data = global_norm_t.getData();

    for (unsigned int idx = 0; idx < lazy_weights.size(); idx++) {
      auto const &w = lazy_weights[idx];

      if (isMixedPrecision()) {
        Tensor scaled_grad =
          w->getGradientRef().clone(TensorDim::DataType::FP32);
        scaled_grad.divide_i(loss_scale);
        global_norm_data[idx] = scaled_grad.l2norm();
      } else {
        global_norm_data[idx] = w->getGradientNorm();
      }
    }
    float global_norm = global_norm_t.l2norm();
    /** apply the gradient with the above global norm */
    for (auto w : lazy_weights) {
      w->clipGradientByGlobalNorm(global_norm);
    }
  }
  /** apply the gradient with the above global norm */
  for (auto w : lazy_weights) {
    lazy_apply_grad_op(*w, iteration, opt);
  }
  nan_count++;

  /** @todo : handle as property : growth_interval : default --> 2000 */
  if (nan_count > 2000) {
    float scale = (*iter_)->getRunContext().getLossScale();
    /** @todo growth_factor : default --> 2.0 */
    float s = scale * 2.0f;
    resetLossScale(s);
    nan_count = 0;
  }

  return true;
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void SubGraphCpu::allocateTensors(ExecutionMode exec_mode_) {
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
     * @todo if model is gradient clipping, we have to add last execution order
     * + 1
     */
    tensor_manager->allocateTensors(
      std::get<3>(backward_iter_end->getExecutionOrder()));
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

  /**
   * @todo for layers which support in-place, both variables and gradients
   * will be shared.
   */
}

std::vector<Var_Grad *>
SubGraphCpu::finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                             const std::vector<Var_Grad *> &prev_inputs) {
  const GraphNode &gnode = *lnode.get();
  std::vector<TensorDim> input_dims;
  input_dims.reserve(prev_inputs.size());
  std::transform(prev_inputs.begin(), prev_inputs.end(),
                 std::back_inserter(input_dims),
                 [](const Var_Grad *vg) { return vg->getDim(); });

  /** finalize the layer and get the final context */
  auto init_context = lnode->finalize(input_dims, getTensorType(), exec_mode);

  /**
   * Request manager for either a pre-allocated output as input or a newly
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used.
   */
  std::vector<std::string> input_names;
  input_names.reserve(prev_inputs.size());
  std::transform(
    prev_inputs.begin(), prev_inputs.end(), std::back_inserter(input_names),
    [](auto const &vg) -> const auto & { return vg->getName(); });
  const std::vector<Var_Grad *> &inputs = tensor_manager->requestInputs(
    gnode, init_context.getInputDimensions(), input_names);

  /** In-Place optimizations */
  /**
   * Request manager for either a pre-allocated input as output or a newly
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used with in-place optimizations.
   */
  auto out_specs = init_context.getOutSpecs();

  /// @note try move inplace control to finalize
  bool shared_var = false, shared_grad = false;

  if (lnode->getInPlaceType() != InPlaceType::NONE && lnode->supportInPlace()) {
    setInplaceSharedMemoryConfigByLayer(lnode, shared_var, shared_grad);
    for (unsigned int i = 0; i < out_specs.size(); ++i) {
      auto &s = out_specs.at(i);
      if (shared_var) {
        s.variable_spec.request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.variable_spec.reference_name = inputs[i]->getName();
          s.variable_spec.dim.setFormat(inputs[i]->getDim().getFormat());
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          s.variable_spec.reference_name = inputs[1]->getName();
          s.variable_spec.dim.setFormat(inputs[1]->getDim().getFormat());
        } else {
          s.variable_spec.reference_name = inputs[0]->getName();
          s.variable_spec.dim.setFormat(inputs[0]->getDim().getFormat());
        }
      }
      if (shared_grad && s.gradient_spec) {
        s.gradient_spec->request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.gradient_spec->reference_name = inputs[i]->getGradientName();
          s.gradient_spec->dim.setFormat(inputs[i]->getDim().getFormat());
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          s.gradient_spec->reference_name = inputs[1]->getGradientName();
          s.gradient_spec->dim.setFormat(inputs[1]->getDim().getFormat());
        } else {
          s.gradient_spec->reference_name = inputs[0]->getGradientName();
          s.gradient_spec->dim.setFormat(inputs[0]->getDim().getFormat());
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

  /// @note below needs to be enabled only for inference mode, but need
  /// decision if we are going to separate inference initialization from
  /// train initialization this might not worth optimize because in general
  /// output of a neuralnet is very small
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
    /// @note below is commented but kept from quick fix to be referenced
    /// for later(#1707)
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
    //   << "shared node must be prior to the dependent node and it should
    //   be
    //   "
    //      "finalized beforehand, shared node name: "
    //   << shared_node_str << " dependent node name: " << lnode->getName();
    // auto num_weight = shared_node->getNumWeights();
    // shared_weight_names.reserve(num_weight);
    // for (auto i = 0u; i < num_weight; ++i) {
    //   shared_weight_names.emplace_back(shared_node->getWeightName(i));
    // }
    // auto &rc = node->getRunContext();

    /// @fixme tensor should be only shared if context explicitly requested
    /// to do so. This has to be added to the part of tensor spec, other
    /// wise it will break many things
    const auto &t_specs = init_context.getTensorsSpec();
    for (auto i = 0u; i < t_specs.size(); ++i) {
      shared_tensor_names.emplace_back(std::get<3>(t_specs.at(i)));
    }

    const auto &w_specs = init_context.getWeightsSpec();
    for (auto i = 0u; i < w_specs.size(); ++i) {
      shared_weight_names.emplace_back(std::get<8>(w_specs.at(i)));
    }
  }
  lnode->setDataType(init_context.getWeightDataType(),
                     init_context.getActivationDataType());
  bool trainable = lnode->getTrainable();
  if (exec_mode == ExecutionMode::INFERENCE)
    trainable = false;
  lnode->configureRunContext(
    // TODO: update weights spec for trainable based on layer trainable prop
    tensor_manager->requestWeights(gnode, init_context.getWeightsSpec(),
                                   trainable, shared_weight_names),
    inputs, outputs,
    tensor_manager->requestTensors(gnode, init_context.getTensorsSpec(),
                                   trainable, shared_tensor_names),
    init_context.getLossScale());

  return outputs;
}

std::vector<Var_Grad *>
SubGraphCpu::refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
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
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used.
   */
  std::vector<std::string> input_names;
  input_names.reserve(prev_inputs.size());
  std::transform(
    prev_inputs.begin(), prev_inputs.end(), std::back_inserter(input_names),
    [](auto const &vg) -> const auto & { return vg->getName(); });
  const std::vector<Var_Grad *> &inputs = tensor_manager->requestInputs(
    gnode, init_context.getInputDimensions(), input_names);

  /** In-Place optimizations */
  /**
   * Request manager for either a pre-allocated input as output or a newly
   * allocated output. This is necessary for manager to know when this
   * output node is going to be used with in-place optimizations.
   */
  auto out_specs = init_context.getOutSpecs();
  /// @note try move inplace control to finalize
  bool shared_var = false, shared_grad = false;
  if (lnode->getInPlaceType() != InPlaceType::NONE) {
    setInplaceSharedMemoryConfigByLayer(lnode, shared_var, shared_grad);
    for (unsigned int i = 0; i < out_specs.size(); ++i) {
      auto &s = out_specs.at(i);
      if (shared_var) {
        s.variable_spec.request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.variable_spec.reference_name = inputs[i]->getName();
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          s.variable_spec.reference_name = inputs[1]->getName();
        } else {
          s.variable_spec.reference_name = inputs[0]->getName();
        }
      }
      if (shared_grad && s.gradient_spec) {
        s.gradient_spec->request_type =
          TensorSpecV2::RequestType::READ_ONLY_VIEW;
        if (lnode->getType() == IdentityLayer::type) {
          s.gradient_spec->reference_name = inputs[i]->getGradientName();
        } else if (lnode->getInPlaceDirection() == InPlaceDirection::RIGHT) {
          // @note With binary inputs, inputs[0] represents the left input
          // tensor while inputs[1] represents the right input tensor. As a
          // result, if the in-place direction is set to right, the in-place
          // memory is assigned to inputs[1].
          s.gradient_spec->reference_name = inputs[1]->getGradientName();
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

  /// @note below needs to be enabled only for inference mode, but need
  /// decision if we are going to separate inference initialization from
  /// train initialization this might not worth optimize because in general
  /// output of a neuralnet is very small
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
    /// @note below is commented but kept from quick fix to be referenced
    /// for later(#1707)
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
    //   << "shared node must be prior to the dependent node and it should
    //   be
    //   "
    //      "finalized beforehand, shared node name: "
    //   << shared_node_str << " dependent node name: " << lnode->getName();
    // auto num_weight = shared_node->getNumWeights();
    // shared_weight_names.reserve(num_weight);
    // for (auto i = 0u; i < num_weight; ++i) {
    //   shared_weight_names.emplace_back(shared_node->getWeightName(i));
    // }
    // auto &rc = node->getRunContext();

    /// @fixme tensor should be only shared if context explicitly requested
    /// to do so. This has to be added to the part of tensor spec, other
    /// wise it will break many things
    const auto &t_specs = init_context.getTensorsSpec();
    for (auto i = 0u; i < t_specs.size(); ++i) {
      shared_tensor_names.emplace_back(std::get<3>(t_specs.at(i)));
    }

    const auto &w_specs = init_context.getWeightsSpec();
    for (auto i = 0u; i < w_specs.size(); ++i) {
      shared_weight_names.emplace_back(std::get<8>(w_specs.at(i)));
    }
  }

  auto weights = lnode->getRunContext().getWeights();
  lnode->configureRunContext(
    // TODO: update weights spec for trainable based on layer trainable prop
    weights, inputs, outputs,
    tensor_manager->requestTensors(gnode, init_context.getTensorsSpec(),
                                   lnode->getTrainable(), shared_tensor_names),
    init_context.getLossScale());

  return outputs;
}

int SubGraphCpu::initialize(ExecutionMode mode,
                            const std::vector<Connection> &model_input_names,
                            const std::vector<Connection> &model_label_names) {

  exec_mode = mode;
  tensor_manager->setExecutionMode(mode);
  /**
   * this contains the map from node name to its input tensor names
   * @note: these input tensors have already been allocated
   */
  std::unordered_map<std::string, std::vector<Var_Grad *>> input_map;

  /** check if the given config of node is of input node */
  auto is_input_node = [](const LayerNode *node) -> bool {
    return node->getInputConnections().empty();
  };

  for (unsigned int idx = 0; idx < subgraph.size(); ++idx) {
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
    if (idx == subgraph.size() - 1)
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

  for (unsigned int idx = 0; idx < subgraph.size(); ++idx) {
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
         * if the gradient is to be clipped by global norm, then the last
         * access is by clipping itself. However, as clipping is not a layer
         * and does not contain any weights, such weights never get assigned
         * gradient_last_access. This is a quick hotfix.
         * TODO: make an independent clipping layer which will execute at
         * the end, and will share ownership of weights which it will clip.
         * This will remove this hot fix, and also remove the checks of if
         * weights require clipping.
         */
        if (tensor_manager->isLastAccess(rc.getWeightGrad(i).getName(),
                                         last_grad_access) ||
            ((rc.isGradientClipByGlobalNorm(i) || rc.isMixedPrecision(i)) &&
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
      for (unsigned int i = 0; i < subgraph.size(); ++i) {
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
      for (unsigned int i = 0; i < subgraph.size(); ++i) {
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
    ml_loge("Backwarding required from layer which doesn't support "
            "backwarding: %s",
            e.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** select weights which would require clipping of the gradients by global
   * norm if any */
  lazy_weights = tensor_manager->getWeights([](const Weight *w) {
    return w->hasGradient() && w->isGradientLastAccess() &&
           (w->isGradientClipByGlobalNorm() || w->isMixedPrecision());
  });

  is_clip_grad = false;
  for (auto w : lazy_weights) {
    if (w->isGradientClipByGlobalNorm()) {
      is_clip_grad = true;
      break;
    }
  }
  return ML_ERROR_NONE;
}

int SubGraphCpu::reinitialize(
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

  for (unsigned int idx = 0; idx < subgraph.size(); ++idx) {
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
    if (idx == subgraph.size() - 1)
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

  for (unsigned int idx = 0; idx < subgraph.size(); ++idx) {
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
         * if the gradient is to be clipped by global norm, then the last
         * access is by clipping itself. However, as clipping is not a layer
         * and does not contain any weights, such weights never get assigned
         * gradient_last_access. This is a quick hotfix.
         * TODO: make an independent clipping layer which will execute at
         * the end, and will share ownership of weights which it will clip.
         * This will remove this hot fix, and also remove the checks of if
         * weights require clipping.
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
      for (unsigned int i = 0; i < subgraph.size(); ++i) {
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
      for (unsigned int i = 0; i < subgraph.size(); ++i) {
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

void SubGraphCpu::setExternalTensors(const std::vector<Tensor> &data,
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

std::vector<Tensor> SubGraphCpu::getOutputTensors() const {
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(output_list.size());

  for (auto const &name : output_list)
    output_tensors.push_back(*tensor_manager->getTensor(name));

  return output_tensors;
}

void SubGraphCpu::LoadTensors(unsigned int order) {
  tensor_manager->LoadTensors(order);
}

bool SubGraphCpu::checkLoadComplete(unsigned int order) {
  return tensor_manager->checkLoadComplete(order);
}

bool SubGraphCpu::checkUnloadComplete(unsigned int order) {
  return tensor_manager->checkUnloadComplete(order);
}

void SubGraphCpu::UnloadTensors(unsigned int order) {
  tensor_manager->UnloadTensors(order);
}

void SubGraphCpu::requestOptimizerVariable(
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

void SubGraphCpu::forwarding_op(std::shared_ptr<LayerNode> node, bool training,
                                bool swap_mode) {
  PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());
  auto f = std::get<0>(node->getExecutionOrder());
  // temperally remain. when we evaluate all for asynch mode, we weill remove
  if (training or (!training and !swap_mode)) {
    tensor_manager->flushCacheExcept(f);
    node->forwarding(training);
  } else {
    /**
     currently, it supports FSU asynch mode for inference. The prcedure of
     FSU is below,
     Prerequests : This function is called node by node at the forwarding
     function in network graph.
     Step 1. If the execution order is the first (f==0) then, it will try to
             load tensors which used at layer 0.
     Step 2. It check whether these tensors from Step 1, then do the
             forwarding of the first node.
     Step 3. Then check the look a head which says how many layer weights need
             to be loaded before running to hide overehad due to FSU,
     Step 4. Try to get the tesors by asking tensors for layers which is done
             by thread pool
     Step 5. Try to release the weights which has execution order less then f.
     Step n. repeat next layer starting with checking the tenosrs are loaded,
             and if it is loaded, then run forwarding. Every time it finishes
             the forwarding, ask load tensors for next n layers.
    **/
    if (f == 0)
      tensor_manager->LoadTensors(f);
    if (tensor_manager->checkLoadComplete(f)) {
      node->forwarding(training);
      ml_logd("Forwarding is done %d : %s", f, node->getName().c_str());
      if (lookahead != 0) {
        if ((f) % (lookahead + 1) == lookahead - 1) {
          ml_logd("request load tensor for %d", f + 1);
          tensor_manager->LoadTensors((f / (lookahead + 1) + 1) *
                                      (lookahead + 1));
        }
      } else {
        tensor_manager->LoadTensors(f);
      }
      if (f != 0)
        tensor_manager->UnloadTensors(f);
    }
  }
}

void SubGraphCpu::incremental_forwarding_op(std::shared_ptr<LayerNode> node,
                                            unsigned int from, unsigned int to,
                                            bool training) {
  PROFILE_MEM_ANNOTATE("Forwarding for layer: " + node->getName());
  auto f = std::get<0>(node->getExecutionOrder());
  tensor_manager->flushCacheExcept(f);
  node->incremental_forwarding(from, to, training);
}

bool SubGraphCpu::backwarding_op(std::shared_ptr<LayerNode> node, int iteration,
                                 std::function<bool(void *userData)> stop_cb,
                                 void *user_data, bool is_grad_opt_mode,
                                 std::shared_ptr<OptimizerWrapped> opt) {

  /**
   * Do not change this order:
   * 1. calcGradient
   * 2. calcDerivative
   * 3. applyGradient
   * 4. gradientClippingOnLastAccess
   */

  flushCacheExcept(std::get<1>(node->getExecutionOrder()));
  PROFILE_MEM_ANNOTATE("CalcGradient: " + node->getName());

  bool apply_gradient = true;
  if (node->getTrainable()) {
    /** If gradient optimization mode, then calculate gradient first */
    if (is_grad_opt_mode)
      node->calcGradient();

    /**
     * If optimization off, or gradient must be applied, then this will be
     * true
     * @todo This apply gradient should be passed to the each weight and later
     * be queried when updating gradient at once. (after moving apply_gradient
     * out of this function)
     *
     */
    // auto &layer = node->getObject();
    // apply_gradient = dynamic_training_opt.checkIfApply(
    //   layer->getWeightsRef(), layer->net_input[0], layer->net_hidden[0],
    //   opt, iteration);

    /** If gradient must be applied and its not gradient mode, calculate
     * gradient
     */
    if (!is_grad_opt_mode && apply_gradient) {
      node->calcGradient();

      RunLayerContext &rc = node->getRunContext();
      if (isMixedPrecision()) {
        for (auto w : rc.getWeights()) {
          if (w->hasGradient())
            if (!w->getGradientRef().isValid())
              return false;
        }
      }
    }
  }

  flushCacheExcept(std::get<2>(node->getExecutionOrder()));
  PROFILE_MEM_ANNOTATE("CalcDerivative: " + node->getName());

  if (stop_cb(user_data)) {
    return true;
  }

  if (node->needsCalcDerivative()) {
    node->calcDerivative();
  }

  flushCacheExcept(std::get<3>(node->getExecutionOrder()));
  PROFILE_MEM_ANNOTATE("ApplyGradient: " + node->getName());

  if (apply_gradient) {
    /// Apply gradient only at the end of the last shared weight access
    applyGradients(node.get(), iteration, opt);
  }
  return true;
}

void SubGraphCpu::lazy_apply_grad_op(Weight &w, int iteration,
                                     std::shared_ptr<OptimizerWrapped> opt) {
  w.calcRegularizationGradient();
  w.calcWeightDecayGradient();
  RunOptimizerContext opt_context(&w, iteration,
                                  opt->getLearningRate(iteration));
  opt->applyGradient(opt_context);
}

void SubGraphCpu::flushCache() { tensor_manager->flushCache(); }

void SubGraphCpu::flushCacheExcept(unsigned int order) {
  tensor_manager->flushCacheExcept(order);
}

} /* namespace nntrainer */
