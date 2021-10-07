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
  Manager() : enable_optimizations(true) {}

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
  ~Manager() = default;

  /**
   * @brief     Create weights with the given spec
   *
   * @param node Graph node to extract node identifiers/info
   * @param weights_spec Specficiation for the weights
   * @param trainable make the weight trainable if true
   *
   * @return created weights list
   */
  std::vector<Weight *>
  requestWeights(const GraphNode &node,
                 const std::vector<Weight::Spec> &weights_spec, bool trainable);

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
    Tensor::Initializer initializer = Tensor::Initializer::NONE);

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
   * @param inputs_name Name of the inputs tensors which for tensor sharing
   *
   * @return created tensors list
   */
  std::vector<Var_Grad *>
  requestOutputs(const GraphNode &node,
                 const std::vector<TensorDim> &outputs_dim,
                 const std::vector<std::string> &inputs_name = {});

  /**
   * @brief     Get all the weights
   *
   * @return    return all the weights
   */
  std::vector<Weight *> getWeights();

  /**
   * @brief Get the Min Max of a tensor execution order
   *
   * @param name name of the tensor
   * @return std::pair<unsigned int, unsigned int>
   */
  std::pair<unsigned int, unsigned int>
  getMinMaxTensorExecutionOrder(const std::string &name);

  /**
   * @brief check if given execution order is the first access
   *
   * @param name tensor name
   * @param current_execution current execution
   * @return bool true if given execution order first access
   */
  bool isFirstAccess(const std::string &name, unsigned current_execution);

  /**
   * @brief check if given execution order is the last access
   *
   * @param name tensor name
   * @param current_execution current execution
   * @return bool ture if given execution order is the last access
   */
  bool isLastAccess(const std::string &name, unsigned current_execution);

  /*
   * @brief   Check if the manager has allocated tensors
   *
   * @return true if tensors allocated, else false
   */
  bool isAllocated() const { return tensor_pool.isAllocated(); }

  /**
   * @brief Set the batch size for the inputs/outputs of the layers
   */
  void setBatchSize(unsigned int batch) {
    /**
     * All the tensors must be deallocated first by the called and then
     * allocated by the caller.
     */
    for (auto &in : inputs_v2)
      in->setBatchSize(batch);
    for (auto &out : outputs_v2)
      out->setBatchSize(batch);
  }

  /**
   * @brief Set the batch size for the given tensor
   *
   * @note this does not works for weights as they are supposed to be
   * independent of batch size.
   */
  void setBatchSize(const std::string &name, unsigned int batch) {
    tensor_pool.setBatchSize(name, batch);
  }

  /**
   * @brief Allocate memory for all the managed tensors
   *
   * @param[in] max_exec_order The maximum order of execution to determine
   * memory layout
   *
   * @note Any requested tensor which is not used inside the max_exec_order is
   * not initialized and will not be allocated. The initialization uses a memory
   * planner to plan the layout of all the tensors which are used at least once
   * before the max_exec_order.
   */
  void allocateTensors(unsigned int max_exec_order_);

  /**
   * @brief Deallocate memory for all the managed tensors
   */
  void deallocateTensors(bool dealloc_weights = false);

  /**
   * @brief Allocate memory for all the managed weights
   *
   * @param[in] max_exec_order The maximum order of execution to determine
   * memory layout
   *
   * @note Any requested tensor which is not used inside the max_exec_order is
   * not initialized and will not be allocated. The initialization uses a memory
   * planner to plan the layout of all the tensors which are used at least once
   * before the max_exec_order.
   *
   * @note this will make requests to the tensor pool and allocate the
   * corresponding weights
   */
  void allocateWeights(unsigned int max_exec_order_);

  /**
   * @brief Deallocate memory for all the weights
   */
  void deallocateWeights();

  /**
   * @brief Set optimizations for manager
   *
   * @param val true to enable, else false
   */
  void setOptimizations(bool val) { enable_optimizations = val; }

  /**
   * @brief Update the dependency on external tensors
   */
  void updateExternalTensors() {
    weight_pool.updateExternalTensors();
    tensor_pool.updateExternalTensors();
  }

  /**
   * @brief Update externally dependent tensors
   *
   * @param name Name of the tensor
   * @param t External tensor
   */
  void setExternalTensor(const std::string &name, const Tensor &t) {
    tensor_pool.setExternalTensor(name, t);
  }

private:
  /** @todo: merge this list to one */
  std::vector<std::unique_ptr<Weight>>
    weights_v2; /**< weights for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    inputs_v2; /**< inputs for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    outputs_v2; /**< outputs for the layers */
  std::vector<std::unique_ptr<Var_Grad>>
    tensors_v2; /**< extra tensors required by the layers */

  TensorPool weight_pool; /**< tensor pool to request tensors */
  TensorPool tensor_pool; /**< tensor pool to request tensors */

  bool enable_optimizations; /**< to enable memory optimizations */

  /**
   * @brief Finalize the given tensor pool
   *
   * @param pool Tensor pool to finalize
   * @param start Start execution order
   * @param end End execution order
   */
  void finalizeTensorPool(TensorPool &pool, unsigned int start,
                          unsigned int end);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
