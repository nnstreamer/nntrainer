// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   manager.h
 * @date   30 Nov 2020
 * @brief  This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 *
 * @details Manager assumes that the layer inouts are being tracked by the
 * manager in the order of the execution. If the order is not maintained, then
 * the optimizations cannot be performed and will result in wrong values.
 */

#ifndef __MANAGER_H__
#define __MANAGER_H__
#ifdef __cplusplus

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <basic_planner.h>
#include <graph_node.h>
#include <tensor_pool.h>
#include <var_grad.h>
#include <weight.h>

namespace nntrainer {

/**
 * @class MMappedMemory
 * @brief Memory Handler, that has mmaped memory with a file descriptor
 */
class MMapedMemory {
public:
  /**
   * @brief Construct a new MMapedMemory object
   *
   * @param size bytesize of the memory chunk
   * @param allocate_fd_ map a shared memory object to a file
   */
  MMapedMemory(size_t size, bool allocate_fd_ = false);

  /**
   * @brief Destroy the MMapedMemory object
   *
   */
  ~MMapedMemory() noexcept;

  /**
   * @brief Construct a new MMapedMemory object (deleted)
   *
   */
  MMapedMemory(const MMapedMemory &) = delete;

  /**
   * @brief Copy assignment operator (deleted)
   *
   */
  MMapedMemory &operator=(const MMapedMemory &) = delete;

  /**
   * @brief Get the File descriptor.
   * Will return -1 except for android
   * @todo make this available for other platforms
   *
   * @return -1 if fd is not allocated (or unabled to allocate)
   */
  int getFd() noexcept { return fd; }

  /**
   * @brief get the size of managed memory
   *
   * @return size_t size
   */
  size_t size() noexcept { return buf_size; }

  /**
   * @brief get Typed buffer from the memory
   *
   * @tparam T Type to specify the buffer. return is reinterpreted to T*
   * @return T* Typed buffer, return nullptr if empty
   */
  template <typename T> T *typedBuffer() noexcept {
    return reinterpret_cast<T *>(buf);
  }

  void *data() noexcept { return typedBuffer<void>(); }

private:
  int fd;           /**< fd to access the shared_memory  */
  void *buf;        /**< buffer object when use_shared_memory */
  size_t buf_size;  /**< buffer size */
  bool allocate_fd; /**< option to choose to allocate an fd */
};

/**
 * @class   Manager
 * @brief   manager of nntrainer
 */
class Manager {

public:
  /**
   * @brief     Constructor of Manager
   */
  Manager(bool enable_gradient_memory_opt_ = true,
          bool enable_derivative_memory_opt_ = false,
          bool enable_activation_memory_opt_ = false,
          bool enable_inference_inout_memory_opt_ = false);

  /**
   * @brief Construct a new Manager object (deleted)
   *
   */
  Manager(const Manager &) = delete;

  /**
   * @brief Copy Assign a new Manager object (deleted)
   *
   */
  Manager &operator=(const Manager &) = delete;

  /**
   * @brief Move Construct a new Manager object
   *
   */
  Manager(Manager &&) noexcept = default;

  /**
   * @brief Move assign a new Manager object
   *
   * @return Manager& reference to newly assign
   */
  Manager &operator=(Manager &&) noexcept = default;

  /**
   * @brief     Destructor of Manager
   */
  ~Manager();

  /**
   * @brief     Add weight to be tracked and updated with nntrainer
   *
   * @param w   Weight to be tracked
   */
  void trackWeight(std::reference_wrapper<Weight> w);

  /**
   * @brief     Add weights to be tracked and updated with nntrainer
   *
   * @param ws  Weights to be tracked
   */
  void trackWeights(std::vector<Weight> &ws);

  /**
   * @brief     Create weights with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param weights_spec Specficiation for the weights
   *
   * @return created weights list
   */
  std::vector<Weight *>
  requestWeights(const GraphNode &node,
                 const std::vector<Weight::Spec> &weights_spec);

  /**
   * @brief     Create tensors with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param tensors_spec Specficiation for the tensors
   *
   * @return created tensors list
   */
  std::vector<Var_Grad *>
  requestTensors(const GraphNode &node,
                 const std::vector<Var_Grad::Spec> &tensors_spec);

  /**
   * @brief     Create tensors with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param tensors_spec Specficiation for the tensors
   *
   * @return created tensors list
   */
  std::vector<Tensor *> requestWeightOptimizerVariables(
    const std::vector<TensorDim> &dims, const std::string &name,
    const TensorLifespan &lifespan,
    Tensor::Initializer initializer = Tensor::Initializer::NONE) {
    auto const &exec_order = weight_pool.getExecutionOrder(name);

    std::vector<Tensor *> ret;
    ret.reserve(dims.size());

    for (unsigned int idx = 0; idx < dims.size(); idx++)
      ret.push_back(tensor_pool.requestTensor(
        dims[idx], exec_order, lifespan, name + ":opt" + std::to_string(idx),
        initializer));

    return ret;
  }

  /**
   * @brief     Create tensors with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param inputs_dim Specficiation for the tensors
   * @param outputs_name Name of the already requested output tensors
   *
   * @return created tensors list
   *
   * @details create Var_Grads to be used as input of GraphNode with the
   * inputs_dim as their spec. If the outputs_name is provided, the returned
   * Var_Grad share tensors with the already allocated Var_Grad for outputs,
   * named with outputs_name. In this case, the input_dim and the shape of the
   * output_tensors must match. If the outputs_name are empty, then new tensors
   * will be allocated.
   */
  std::vector<Var_Grad *>
  requestInputs(const GraphNode &node, const std::vector<TensorDim> &inputs_dim,
                const std::vector<std::string> &outputs_name = {});

  /**
   * @brief     Create tensors with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param outputs_dim Specficiation for the tensors
   *
   * @return created tensors list
   */
  std::vector<Var_Grad *>
  requestOutputs(const GraphNode &node,
                 const std::vector<TensorDim> &outputs_spec);

  /**
   * @brief     Create tensors with the given spec and name
   *
   * @param node Graph node to extract node identifiers/info
   * @param tensors_dim Specficiation for the tensors
   *
   * @return created tensors list
   */
  std::vector<Var_Grad *>
  requestAllocatedOutputsAsInputs(const GraphNode &node,
                                  const std::vector<TensorDim> &tensors_dim,
                                  const std::vector<std::string> &outputs_name);

  /**
   * @brief     Get all the weights
   *
   * @return    return all the weights
   */
  std::vector<Weight *> getWeights();

  /**
   * @brief     Get weights tracked with nntrainer
   *
   * @retval    list of weight references
   */
  std::vector<std::vector<std::reference_wrapper<Weight>>> getWeightRefs() {
    return weights;
  }

  /**
   * @brief Enable gradient memory sharing based optimization
   * @param opt True to enable, else false
   */
  void setGradientMemoryOptimization(bool opt) {
    enable_gradient_memory_opt = opt;
  }

  /**
   * @brief Enable derivative memory sharing based optimization
   * @param opt True to enable, else false
   */
  void setDerivativeMemoryOptimization(bool opt) {
    enable_derivative_memory_opt = opt;
  }

  /**
   * @brief Enable derivative memory sharing based optimization
   * @param opt True to enable, else false
   */
  void setInPlaceActivationOptimization(bool opt) {
    if (opt)
      throw exception::not_supported(
        "Inplace activation optimization is temporarily disabled");
    enable_activation_memory_opt = opt;
  }

  /**
   * @brief Enable inout memory sharing based optimization for inference
   * @param opt True to enable, else false
   */
  void setInferenceInOutMemoryOptimization(bool opt) {
    if (opt)
      throw exception::not_supported(
        "Inference memory optimization is temporarily disabled");
    enable_inference_inout_memory_opt = opt;
  }

  /**
   * @brief Allocate and initialize the weight variable
   * @note This only allocates weights and does not handle training related
   * memory for weights
   */
  void initializeWeights(unsigned int max_exec_order);

  /**
   * @brief Reset the manager state
   * @note The tensors assigned to the layers are not reset. They will be
   * automatically reset once the model is initialized again.
   */
  void reset() {
    deallocateTensors(true);

    deinitializeTensors();
    weights_initialized = false;

    weight_mmaped_memory.reset();
    grad_mmaped_memory.reset();

    /** reset model registered variables */
    total_weight_size = 0;
    total_grad_size = 0;
    max_grad_size = 0;
    max_derivative_size = 0;
    max_shared_inout = 0;

    in_outs.clear();
    weights.clear();
    is_act_type.clear();
    is_rnn_type.clear();
    is_flat_type.clear();
  }

  /**
   * @brief Track the inputs of the layer
   * @param[in] layer_type Type of the layer
   * @param[in] layer_name Name of the layer
   * @param[in] input_dim Dimension of the input for the layer
   * @param[in] output_dim Dimension of the output for the layer (optional)
   * @retval created objects for input of the layer
   * @note Manager is kept independent from the layer object itself
   * @note This function only allocates variables using the input_dim of the
   * layer. The output dimension is for optimization purposes.
   */
  std::vector<std::shared_ptr<Var_Grad>> &
  trackLayerInputs(const std::string &layer_type, const std::string &layer_name,
                   const std::vector<TensorDim> &input_dim,
                   const std::vector<TensorDim> &output_dim = {});

  /**
   * @brief Track the ouputs of the layer
   * @param[in] layer_type Type of the layer
   * @param[in] layer_name Name of the layer
   * @param[in] output_dim Dimension of the output for the layer
   * @param[in] input_dim Dimension of the input for the layer (optional)
   * @retval created objects for output of the layer
   * @note Manager is kept independent from the layer object itself
   * @note This function only allocates variables using the output_dim of the
   * layer. The input dimension is for optimization purposes.
   */
  std::vector<std::shared_ptr<Var_Grad>> &
  trackLayerOutputs(const std::string &layer_type,
                    const std::string &layer_name,
                    const std::vector<TensorDim> &output_dim,
                    const std::vector<TensorDim> &input_dim = {});
  /**
   * @brief Track the inputs/ouputs of the layer
   * @param[in] layer_name Name of the layer
   * @note Manager is kept independent from the layer object itself
   */
  void untrackLayerInOuts(const std::string &layer_name);

  /**
   * @brief Initialize the all the requested tensors
   *
   * @param[in] training If model will be training or not
   * @param[in] max_exec_order The maximum order of execution to determine
   * memory layout
   *
   * @note Any requested tensor which is not used inside the max_exec_order is
   * not initialized and will not be allocated. The initialization uses a memory
   * planner to plan the layout of all the tensors which are used at least once
   * before the max_exec_order.
   */
  void initializeTensors(bool training, unsigned int max_exec_order);

  /**
   * @brief   Check if the manager has allocated tensors
   *
   * @return true if tensors allocated, else false
   */
  bool isAllocated() const { return tensors_allocated; }

  /**
   * @brief Set the batch size for the inputs/outputs of the layers
   */
  void setBatchSize(unsigned int batch) {
    if (!in_outs.empty() && !in_outs[0].empty()) {
      unsigned int prev_batch = in_outs[0][0]->getDim().batch();
      max_derivative_size /= prev_batch;
      max_shared_inout /= prev_batch;
      max_derivative_size *= batch;
      max_shared_inout *= batch;
    }

    /**
     * All the tensors must be deallocated first by the called and then
     * allocated by the caller.
     */

    for (auto &in_out : in_outs)
      for (auto &vg : in_out)
        vg->setBatchSize(batch);

    for (auto &in : inputs_v2)
      in->setBatchSize(batch);
    for (auto &out : outputs_v2)
      out->setBatchSize(batch);
  }

  /**
   * @brief Set the batch size for the given tensor
   *
   * @note this only works for tensors_v2 for now
   */
  void setBatchSize(const std::string &name, unsigned int batch) {
    tensor_pool.setBatchSize(name, batch);
  }

  /**
   * @brief Allocate memory for all the managed tensors
   */
  void allocateTensors() {
    if (!weights_allocated)
      allocateWeights();

    if (!tensors_allocated) {
      tensor_pool.finalize(BasicPlanner(), 0, max_exec_order);
      if (model_training)
        allocateGradients();
      allocateInOuts();
      if (model_training)
        allocateDerivatives();

      if (tensor_pool.minMemoryRequirement() > 0)
        tensor_pool.allocate();
      tensors_allocated = true;
    }
  }

  /**
   * @brief Deallocate memory for all the managed tensors
   */
  void deallocateTensors(bool dealloc_weights = false) {
    if (dealloc_weights and weights_allocated)
      deallocateWeights();

    if (tensors_allocated) {
      if (model_training)
        deallocateGradients();
      deallocateInOuts();
      if (model_training)
        deallocateDerivatives();

      tensor_pool.deallocate();
      tensors_allocated = false;
    }
  }

  /**
   * @brief Allocate memory for all the managed weights
   */
  void allocateWeights();

  /**
   * @brief Deallocate memory for all the weights
   */
  void deallocateWeights();

private:
  std::vector<std::unique_ptr<Weight>>
    weights_v2; /**< weights for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    inputs_v2; /**< inputs for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    outputs_v2; /**< outputs for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    tensors_v2; /**< extra tensors required by the layers */

  /** @todo: combine the list of the weights/var_grad to a common list */
  // std::vector<std::unique_ptr<Var_Grad>> tensors; /**< inputs/outputs/tensors
  // for the network */

  /** TODO: kept for now, possibly remove this after for offloading is
   * implemented */
  std::unordered_map<std::string, std::vector<unsigned int>>
    tensor_exec_order; /**< stores the order/location at which a given tensor is
                          going to be used when the network is forwarded and
                          backwarded */

  std::unordered_map<std::string, TensorLifespan>
    tensor_lifespan_map; /**< map from tensor name to its lifespan */
  std::unordered_map<std::string, int>
    tensor_token_map; /**< map from tensor to its memory token */

  std::unordered_map<std::string, int>
    name_map;                  /**< map from output name to its location */
  unsigned int max_exec_order; /**< max execution for a node */

  TensorPool weight_pool; /**< tensor pool to request tensors */
  TensorPool tensor_pool; /**< tensor pool to request tensors */

  /**< Weights of all the layer in the model to be managed */
  std::vector<std::vector<std::reference_wrapper<Weight>>> weights;

  unsigned int total_weight_size; /**< total weight size */
  unsigned int total_grad_size;   /**< total weight size */
  unsigned int max_grad_size; /**< max trainable weight required by a layer */
  unsigned int max_derivative_size; /**< max derivative required by a layer */
  unsigned int max_shared_inout;    /**< max memory for in/outs for inference */

  bool weights_initialized; /**< track if weights have been initialized */
  bool tensors_initialized; /**< track if other tensors have been initialized */
  bool weights_allocated;   /**< track if weights have been allocated */
  bool tensors_allocated;   /**< track if other tensors have been allocated */
  bool model_training;      /**< track if the model is in training mode */

  /**< Inputs/outputs of all the layer in the model */
  std::vector<std::vector<std::shared_ptr<Var_Grad>>> in_outs;
  std::vector<bool> is_act_type;
  std::vector<bool> is_rnn_type;
  std::vector<bool> is_flat_type;
  Tensor
    shared_grad; /**< Shared tensor containing memory for weight gradients */
  Tensor shared_inout; /**< Shared tensor containing memory for input and
                          outputs for inference */
  Tensor shared_deriv; /**< Shared tensor containing memory for input and output
                          derivatives */

  /**< Optimization related */
  bool enable_gradient_memory_opt; /**< share memory among all the gradients */
  bool enable_derivative_memory_opt; /**< share memory among all the derivative
                                        and output of the next layer */
  bool enable_activation_memory_opt; /**< Let activation layer work in-place
                                        without allocating output layer for
                                        itself */
  bool enable_inference_inout_memory_opt; /**< Use shared memory for inputs and
                                             outputs of all the layers in
                                             inference mode */

  /**< shared memory related */
  bool use_shared_memory; /**< uses shared memory object which is owned by
                             manager */
  std::unique_ptr<MMapedMemory> weight_mmaped_memory;
  std::unique_ptr<MMapedMemory> grad_mmaped_memory;

  /** Alloc function definition */
  using AllocFunc = std::function<Tensor(const TensorDim &, unsigned int)>;

  /**
   * @brief Track the inputs/ouputs of the layer
   * @param[in] layer_type Type of the layer
   * @param[in] layer_name Name of the layer
   * @param[in] inout_dim Dimension of the input/output for the layer
   * @retval created objects for input/output of the layer
   * @note Manager is kept independent from the layer object itself
   */
  std::vector<std::shared_ptr<Var_Grad>> &
  trackLayerInOuts(const std::string &layer_type, const std::string &layer_name,
                   const std::vector<TensorDim> &inout_dim);
  /**
   * @brief UnTrack the inputs/ouputs of the layer
   * @param[in] var_name Name of the variable
   */
  void untrackVariable(const std::string &var_name);

  /**
   * @brief Allocate and initialize the weight gradients
   * @note This only allocates weight's gradients and assumes that weights are
   * pre-allocated.
   */
  void initializeGradients();

  /**
   * @brief Get helper allocator function to use for weight or gradient
   * @param[in] is_weight true if weight, else false meaning its gradient
   */
  AllocFunc getAllocFunc(bool is_weight);

  /**
   * @brief Allocate memory for all the managed gradients
   */
  void allocateGradients();

  /**
   * @brief Allocate memory for all the managed layers inputs and outputs
   */
  void allocateInOuts();

  /**
   * @brief Allocate memory for all the managed layer derivatives
   */
  void allocateDerivatives();

  /**
   * @brief Deallocate memory for all the gradients of the weights
   *
   */
  void deallocateGradients();

  /**
   * @brief Deallocate memory for all the input and output tensors
   *
   */
  void deallocateInOuts();

  /**
   * @brief Deallocate memory for all the inputs and outputs derivative tensors
   *
   */
  void deallocateDerivatives();

  /**
   * @brief Deinitialize the tensors
   */
  void deinitializeTensors();

  /**
   * @brief Initialize the tensors for inference mode
   */
  void initializeTensorsInference(unsigned int);

  /**
   * @brief Initialize the tensors for training mode
   */
  void initializeTensorsTrain(unsigned int);

  /**
   * @brief     Create tensors with the given spec
   *
   * @param w   node Graph node to extract node identifiers/info
   * @param w   create tensors list
   * @param layer_objs_list list to store the created tensors
   */
  template <typename T>
  std::vector<T *>
  requestTensors(const GraphNode &node,
                 const std::vector<typename T::Spec> &tensors_spec,
                 std::vector<std::unique_ptr<T>> &layer_objs_list) {
    std::vector<T *> ret;
    size_t current_size = layer_objs_list.size();

    for (auto const &ts : std::as_const(tensors_spec)) {
      layer_objs_list.emplace_back(std::make_unique<T>(ts));
      auto const &ts_name = layer_objs_list.back()->getName();

      if (tensor_exec_order.find(ts_name) != tensor_exec_order.end())
        throw std::invalid_argument("Requesting tensor " + ts_name +
                                    " with same name");

      tensor_exec_order[ts_name] = {};
      name_map[ts_name] = layer_objs_list.size() - 1;
    }

    std::transform(layer_objs_list.begin() + current_size,
                   layer_objs_list.end(), std::back_inserter(ret),
                   [](auto const &elem) { return elem.get(); });

    return ret;
  }

  /**
   * @brief     Expand the lifespan of the tensor with the given name
   *
   * @param name The name of the tensor
   * @param lifespan The lifespan to be expanded to
   */
  inline void expandLifespan(const std::string &name, TensorLifespan lifespan);

  /**
   * @brief     Get validity for the given tensor
   *
   * @param name Name of the tensor
   * @return validity for the given tensor
   * @details the validity will be created using the lifespan and execution
   * order
   */
  std::pair<unsigned int, unsigned int> getValidity(const std::string &name);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
