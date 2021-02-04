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
 */

#ifndef __MANAGER_H__
#define __MANAGER_H__
#ifdef __cplusplus

#include <functional>
#include <memory>
#include <vector>

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

  ~MMapedMemory() noexcept;

  MMapedMemory(const MMapedMemory &) = delete;

  MMapedMemory &operator=(const MMapedMemory &) = delete;

  /**
   * @brief Get the File descriptor.
   * Will return -1 except for android
   * @todo make this available for other platforms
   *
   * @return -1 if fd is not allocated (or unabled to allocate)
   */
  int getFd() noexcept { return fd; }

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
          bool enable_derivative_memory_opt_ = true,
          bool enable_activation_memory_opt_ = true,
          bool enable_inference_inout_memory_opt_ = true);

  Manager(const Manager &) = default;

  Manager &operator=(const Manager &) = default;

  Manager(Manager &&) noexcept = default;

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
    enable_activation_memory_opt = opt;
  }

  /**
   * @brief Enable inout memory sharing based optimization for inference
   * @param opt True to enable, else false
   */
  void setInferenceInOutMemoryOptimization(bool opt) {
    enable_inference_inout_memory_opt = opt;
  }

  /**
   * @brief Allocate and initialize the weight variable
   * @note This only allocates weights and does not handle training related
   * memory for weights
   */
  void initializeWeights();

  /**
   * @brief Reset the manager state
   */
  void reset() {
    weights.clear();
    max_grad_size = 0;
    total_weight_size = 0;
    total_grad_size = 0;
    max_shared_inout = 0;
    weight_mmaped_memory.reset();
    grad_mmaped_memory.reset();
    in_outs.clear();
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
   * @brief Initialize the inputs/outputs/derivatives/gradients for the layers
   * @param[in] trainable If true, initialize derivates/gradients, else, do not.
   * @note The memory allocation strategy varies based on the trainable. The
   * memory allocated for inference mode is not compatible with training, and
   * will require full allocation than reusing memory allocated with inference
   * mode.
   */
  void initializeTensors(bool trainable);

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
     * All the tensors must be deallocated first and then allocated.
     * Deallocating and allocating tensors one by one can potentially lead to
     * high requirement of the peak memory requirement.
     */
    deallocateInOuts();
    deallocateDerivatives();

    for (auto &in_out : in_outs)
      for (auto &vg : in_out)
        vg->setBatchSize(batch);

    allocateInOuts();
    allocateDerivatives();
  }

  /**
   * @brief Allocate memory for all the managed tensors
   */
  void allocateTensors() {
    /// Weights are allocated while initializing
    if (!weights_initialized)
      allocateWeights();
    allocateGradients();
    allocateInOuts();
    allocateDerivatives();
  }

  /**
   * @brief Deallocate memory for all the managed tensors
   */
  void deallocateTensors(bool dealloc_weights = false) {
    if (dealloc_weights)
      deallocateWeights();

    deallocateGradients();
    deallocateInOuts();
    deallocateDerivatives();
  }

  /**
   * @brief Allocate memory for all the managed weights
   */
  void allocateWeights();

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
   * @brief Deallocate memory for all the weights
   */
  void deallocateWeights();

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

private:
  // TODO: ensure that names of these weights are unique
  /**< Weights of all the layer in the model to be managed */
  std::vector<std::vector<std::reference_wrapper<Weight>>> weights;

  unsigned int total_weight_size; /**< total weight size */
  unsigned int total_grad_size;   /**< total weight size */
  unsigned int max_grad_size; /**< max trainable weight required by a layer */
  unsigned int max_derivative_size; /**< max derivative required by a layer */
  unsigned int max_shared_inout;    /**< max memory for in/outs for inference */
  bool weights_initialized; /**< track if weights have been initialized */

  /**< Inputs/outputs of all the layer in the model */
  std::vector<std::vector<std::shared_ptr<Var_Grad>>> in_outs;
  std::vector<bool> is_act_type;
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
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
