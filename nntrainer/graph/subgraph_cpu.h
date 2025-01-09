// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>
 *
 * @file    subgraph_cpu.h
 * @date    07 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is a Network SubGraph Class for Neural Network
 *
 */

#ifndef __SUBGRAPH_CPU_H__
#define __SUBGRAPH_CPU_H__
#ifdef __cplusplus

#include <subgraph_base.h>

namespace nntrainer {

using ExecutionMode = ml::train::ExecutionMode;
using ExecutionOrder = GraphNode::ExecutionOrder;
using PrintPreset = LayerNode::PrintPreset;

class Connection;

/**
 * @class   NeuralNetwork SubGraph Class for CPU
 * @brief   NeuralNetwork SubGraph Class which manage layers
 */
class SubGraphCpu : public SubGraphBase {

public:
  /**
   * @brief     Constructor of NeuralNetwork SubGraph Class
   */
  SubGraphCpu(std::shared_ptr<Manager> tm, unsigned int lookahead_ = 0) :
    SubGraphBase(tm, lookahead_) {}

  /**
   * @brief   Destructor of the NeuralNetwork SubGraph class
   *
   */
  ~SubGraphCpu() = default;

  /**
   * @brief     Compile the subgraph
   * @param[in] loss_type loss for the subgraph
   * returns ML_ERROR_NONE on success, error on failure
   */
  int compile(const std::string &loss_type) override;

  /**
   * @brief     set batch size
   * @param[in] batch size
   */
  void setBatchSize(unsigned int batch_size) override;

  /**
   * @brief try apply gradient if possible
   * @note if it is not the last of the gradient access, this is noop
   * @note if the gradient is to be clipped by norm, this is noop
   *
   * @param node node to try apply gradient
   * @param iteration iteration where the applyGradients is called
   * @param opt shared ptr of the optimizer used for applyGradients (opt is
   * passed from neuralnetwork)
   */
  void applyGradients(LayerNode *node, int iteration,
                      std::shared_ptr<OptimizerWrapped> opt) override;

  /**
   * @brief     forwarding network subgraph
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors forwarding(
    bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool swap_mode = false) override;

  /**
   * @brief     forwarding network subgraph
   * @param[in] from start step
   * @param[in] to end step
   * @param[in] training true if forwarding is on training
   * @retval output tensors
   */
  sharedConstTensors incremental_forwarding(
    unsigned int from, unsigned int to, bool training = false,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr) override;

  /**
   * @brief     backwarding the network subgraph
   * @param[in] iteration current iteration number
   * @param[in] stop_cb callback function which return stop condition
   * @param[in] user_data user data used for backwarding
   * @param[in] is_grad_opt_mode flag to designate grad_opt_mode (passed from
   *            neuralnet)
   * @param[in] opt shared ptr of the optimizer used for applyGradients (opt is
   *            passed from neuralnetwork)
   * @retval ret it is false then the gradient has NaN valude in mixed precision
   *         training. If it is, then we need to control the loss scale factor
   *         and compute again the derivatives.
   */
  bool backwarding(
    int iteration,
    std::function<bool(void *userdata)> stop_cb =
      [](void *user_data) { return false; },
    void *user_data = nullptr, bool is_grad_opt_mode = false,
    std::shared_ptr<OptimizerWrapped> opt = nullptr) override;

  /**
   * @brief initialize network subgraph
   *
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @return int ML_ERROR_NONE if successful
   */
  int initialize(
    ExecutionMode mode = ExecutionMode::TRAIN,
    const std::vector<Connection> &model_input_names = {},
    const std::vector<Connection> &model_label_names = {}) override;

  /**
   * @brief reinitialize network subgraph
   *
   * @param model_input_names model input connection if empty list given, all of
   * node that can be inputs will be identified in the sort order
   * @param model_label_names model label names if empty list given, all of node
   * that can be labels will be identified in the sort order
   * @return int ML_ERROR_NONE if successful
   */
  int reinitialize(
    const std::vector<Connection> &model_input_names = {},
    const std::vector<Connection> &model_label_names = {}) override;

  /**
   * @brief Create run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   */
  std::vector<Var_Grad *>
  finalizeContext(const std::shared_ptr<LayerNode> &lnode,
                  const std::vector<Var_Grad *> &prev_inputs) override;

  /**
   * @brief Recreate run layer context from the given init layer context
   *
   * @param lnode layer node to finalize and set run context
   * @param prev_inputs previous input information
   */
  std::vector<Var_Grad *>
  refinalizeContext(const std::shared_ptr<LayerNode> &lnode,
                    const std::vector<Var_Grad *> &prev_inputs) override;

  /** Interface for manager */

  /**
   * @brief Allocate memory for all the managed tensors in the subgraph
   *
   * @param[in] training If true, initialize derivates/gradients, else, do not.
   */
  void allocateTensors(ExecutionMode exec_mode_) override;

  /**
   * @brief Deallocate memory for all the managed tensors
   */
  void deallocateTensors(bool dealloc_weights = false) override {
    tensor_manager->deallocateTensors(dealloc_weights);
  }

  /**
   * @brief Allocate memory for all the managed weights
   */
  void allocateWeights(bool init = true) override {
    unsigned int max_exec_order =
      std::get<3>(backward_iter_end->getExecutionOrder());

    if (exec_mode == ExecutionMode::INFERENCE)
      max_exec_order = std::get<0>(forward_iter_end->getExecutionOrder());
    tensor_manager->allocateWeights(max_exec_order, init);
  }

  /**
   * @brief Deallocate memory for all the weights
   */
  void deallocateWeights() override { tensor_manager->deallocateWeights(); }

  /**
   * @brief     Enable the memory optimizations for the network
   *
   * @param val true to enable, else false
   */
  void setMemoryOptimizations(bool val) override {
    tensor_manager->setOptimizations(val);
    optimize_memory = val;
  }

  /**
   * @brief     Create optimizer variable for every weights
   *
   * @param cb  Call back function which will return vector of dimension
   * @param request_only_trainable true when only request trainable weight
   */
  void requestOptimizerVariable(
    std::function<std::vector<TensorDim>(const TensorDim &)> cb,
    bool request_only_trainable = true) override;

  /**
   * @brief Get the Output Tensors list for the subgraph
   *
   * @return std::vector<Tensor> List of output tensors
   * @note this tensor list is analogous to the label list
   */
  std::vector<Tensor> getOutputTensors() const override;

  /**
   * @brief Load data of order to the device
   *
   * @param order execution order
   */
  void LoadTensors(const unsigned int order) override;

  /**
   * @brief check data of order is loaded
   *
   * @param order execution order
   */
  bool checkLoadComplete(const unsigned int order) override;

  /**
   * @brief check data of order is Unloaded
   *
   * @param order execution order
   */
  bool checkUnloadComplete(const unsigned int order) override;

  /**
   * @brief Load data of order to the device
   *
   * @param order execution order
   */
  void UnloadTensors(const unsigned int order) override;

  inline static const std::string type = "subgraph_cpu";

private:
  /**
   * @brief Set external data to the given tensors with name
   *
   * @param data External data
   * @param names Names of the tensor to set the data to
   */
  void setExternalTensors(const std::vector<Tensor> &data,
                          const std::vector<std::string> names) override;

  /**
   * @brief forwarding_op function
   */
  void forwarding_op(std::shared_ptr<LayerNode> node, bool training,
                     bool swap_mode = false);
  /**
   * @brief forwarding_op function
   */
  void incremental_forwarding_op(std::shared_ptr<LayerNode> node,
                                 unsigned int from, unsigned int to,
                                 bool training);
  /**
   * @brief backwarding_op
   * @param[in] iteration current iteration number
   * @param[in] stop_cb callback function which return stop condition
   * @param[in] user_data user data used for backwarding
   * @param[in] is_grad_opt_mode flag to designate grad_opt_mode (passed from
   *            neuralnet)
   * @param[in] opt shared ptr of the optimizer used for applyGradients (opt is
   *            passed from neuralnetwork)
   * @retval ret it is false then the gradient has NaN valude in mixed precision
   *         training. If it is, then we need to control the loss scale factor
   *         and compute again the derivatives.
   */
  bool backwarding_op(std::shared_ptr<LayerNode> node, int iteration,
                      std::function<bool(void *userData)> stop_cb,
                      void *user_data, bool is_grad_opt_mode,
                      std::shared_ptr<OptimizerWrapped> opt);
  /**
   * @brief backwarding_op
   * @param[in] iteration current iteration number
   * @param[in] opt shared ptr of the optimizer used for applyGradients (opt is
   *            passed from neuralnetwork)
   */
  void lazy_apply_grad_op(Weight &w, int iteration,
                          std::shared_ptr<OptimizerWrapped> opt);

  /**
   * @brief Flush data to the device
   *
   */
  void flushCache();

  /**
   * @brief Flush data to the device except order
   *
   * @param order except execution order
   */
  void flushCacheExcept(const unsigned int order);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SUBGRAPH_CPU_H__ */
