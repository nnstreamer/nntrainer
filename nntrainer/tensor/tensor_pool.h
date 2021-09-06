// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tensor_pool.h
 * @date   18 Aug 2021
 * @brief  This is TensorPool for all requested tensors
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_POOL_H__
#define __TENSOR_POOL_H__
#ifdef __cplusplus

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <memory_pool.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace nntrainer {

/**
 * @class   TensorPool
 * @brief   tensor pool of nntrainer
 */
class TensorPool {

public:
  /**
   * @brief     Constructor of TensorPool
   */
  TensorPool() : mem_pool() {}

  /**
   * @brief     Destructor of TensorPool
   */
  ~TensorPool() = default;

  /**
   * @brief     Request tensor with the given spec
   *
   * @param dim Tensor dimensions
   * @param exec_order The execution orders for this tensors
   * @param lifespan Lifespan of this tensor
   * @param name Name of this tensor
   * @param init Initializer of the tensor
   *
   * @return ptr to the created tensor
   *
   * @note returns empty tensor which will be filled when allocate is called.
   * @note we assume that the caller checks if the exec_order and lifespan are
   * compatible.
   */
  Tensor *
  requestTensor(const TensorDim &dim,
                const std::vector<unsigned int> &exec_order,
                TensorLifespan lifespan, const std::string &name,
                const Tensor::Initializer &init = Tensor::Initializer::NONE);

  /**
   * @brief     Request tensor with the given name which will be allocated
   * externally
   *
   * @param dim Tensor dimensions
   * @param name Name of this tensor
   * @param init Initializer of the tensor
   *
   * @return ptr to the created tensor
   *
   * @note returns empty tensor which must be filled by the caller before use.
   */
  Tensor *requestExternallyAllocateTensor(
    const TensorDim &dim, const std::string &name,
    const Tensor::Initializer &init = Tensor::Initializer::NONE);

  /**
   * @brief     Request tensor which has been already requested with the given
   * spec
   *
   * @param dim Tensor dimensions
   * @param exec_order The execution orders for this tensors
   * @param lifespan Lifespan of this tensor
   * @param name Name of this tensor
   * @param shared_name Name of the preallocated tensor
   * @param init Initializer of the tensor
   *
   * @return ptr to the tensor
   *
   * @note returns empty tensor which will be filled when allocate is called.
   * @note we assume that the caller checks if the exec_order and lifespan are
   * compatible.
   *
   * @note This interface is separated from requestTensor to reduce bugs related
   * to unintentional tensor sharing.
   */
  Tensor *requestPrerequestedTensor(
    const TensorDim &dim, const std::vector<unsigned int> &exec_order,
    TensorLifespan lifespan, const std::string &name,
    const std::string &shared_name,
    const Tensor::Initializer &init = Tensor::Initializer::NONE);

  /**
   * @brief finalize the requested tensors
   * @param planner planner to layout the tensor memories
   * @param start_order start value for the order_exec (inclusive)
   * @param end_order end value for the order_exec (inclusive)
   *
   * @details finalize the requested tensors, request memory for them and plan
   * layout for their allocations.
   */
  void finalize(const MemoryPlanner &planner, unsigned int start_order,
                unsigned int end_order);

  /**
   * @brief Set the batch size for the inputs/outputs of the layers
   */
  void setBatchSize(const std::string &name, unsigned int batch);

  /**
   * @brief Allocate memory for all the managed tensors
   */
  void allocate();

  /**
   * @brief Deallocate memory for all the managed tensors
   */
  void deallocate();

  /**
   * @brief     Expand the lifespan of the tensor with the given name
   *
   * @param name The name of the tensor
   * @param lifespan The lifespan to be expanded to
   */
  void expand_lifespan(const std::string &name, TensorLifespan lifespan);

  /**
   * @brief     Expand the execution order of the tensor with the given name
   *
   * @param name The name of the tensor
   * @param exec_order The execution orders
   */
  void expand_lifespan(const std::string &name,
                       const std::vector<unsigned int> &exec_order);

  /**
   * @brief     Get execution order for the given tensor
   *
   * @return The execution order of the tensor
   */
  const std::vector<unsigned int> &getExecutionOrder(const std::string &name) {
    return pool[name_map.at(name)].exec_order;
  }

  /**
   * @brief Get the maximum real memory requirement
   *
   * @return The real memory requirement with this strategy in bytes
   */
  size_t size() { return mem_pool.size(); }

  /**
   * @brief Get the minimum theoretical memory requirement
   *
   * @return The theoretical memory requirement with this strategy in bytes
   */
  size_t minMemoryRequirement() { return mem_pool.minMemoryRequirement(); }

  /**
   * @brief Is the tensor pool allocated
   *
   * @return true if the tensors are allocated, else false
   */
  bool isAllocated() const { return mem_pool.isAllocated(); }

  /**
   * @brief Get the tensor of the given name
   *
   * @return ptr to the tensor with the given
   * @throws if no tensor is found with the given name
   */
  Tensor *getTensor(const std::string &name) {
    return pool[name_map.at(name)].tensor.get();
  }

  /**
   * @brief Update externally dependent tensors
   *
   * @param name Name of the tensor
   * @param t External tensor
   *
   * @note Update externally dependent tensors data ptrs from their parents
   */
  void setExternalTensor(const std::string &name, const Tensor &t) {
    auto &spec = pool[name_map.at(name)];
    if (spec.lifespan != TensorLifespan::ZERO_LIFESPAN)
      throw std::invalid_argument(
        "Cannot set external tensor for non-zero lifespan");

    spec.tensor->setData(t.getData());
  }

  /**
   * @brief Update externally dependent tensors
   *
   * @note Update externally dependent tensors data ptrs from their parents
   */
  void updateExternalTensors() {
    for (auto &spec : pool)
      if (spec.dependent)
        spec.tensor->setData(pool[spec.token].tensor->getData());
  }

private:
  /**
   * @brief Spec for storing each request of tensor from tensor pool
   * @todo move tensor initialization from tensor class to requestSpec
   */
  struct requestSpec {
    std::unique_ptr<Tensor> tensor;       /**< tensor object itself */
    std::vector<unsigned int> exec_order; /**< tensor exec order list */
    TensorLifespan lifespan;              /**< tensor lifespan */
    unsigned int token;                   /**< tensor memory token */
    bool dependent; /**< if dependent on another tensor for memory */
  };

  /**
   * note: unordered_map is not directly used for pool to ensure initialization
   * of weights
   */
  std::vector<requestSpec> pool; /**< list of requested tensors */
  std::unordered_map<std::string, unsigned int>
    name_map;          /**< indexing of requested tensors */
  MemoryPool mem_pool; /**< memory pool for the tensors */

  /**
   * @brief     Check if the lifespan leads to long term valitidy
   *
   * @param lifespan Lifespan for the tensor
   *
   * @return true if the tensor should be valid for long term, else false
   */
  bool isTensorLongTerm(const TensorLifespan &lifespan);
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_POOL_H__ */
