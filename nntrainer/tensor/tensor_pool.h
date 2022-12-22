// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tensor_pool.h
 * @date   18 Aug 2021
 * @brief  This is TensorPool for all requested tensors
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_POOL_H__
#define __TENSOR_POOL_H__
#ifdef __cplusplus

#include <functional>
#include <limits>
#include <memory>
#include <unordered_map>
#include <variant>
#include <vector>

#include <cache_pool.h>
#include <tensor.h>
#include <tensor_wrap_specs.h>

namespace nntrainer {

/**
 * @class   TensorPool
 * @brief   tensor pool of nntrainer
 */
class TensorPool {

public:
  static constexpr unsigned PERSIST_END_ORDER =
    std::numeric_limits<unsigned>::max();
  /**
   * @brief     Constructor of TensorPool
   */
  TensorPool() : mem_pool(std::make_unique<MemoryPool>()) {}

  /**
   * @brief     Constructor of TensorPool
   */
  TensorPool(bool enable_swap, const std::string &swap_path = "",
             const std::string &swap_name = "") {
    if (enable_swap)
      mem_pool = std::make_unique<CachePool>(swap_path, swap_name);
    else
      mem_pool = std::make_unique<MemoryPool>();
  }

  /**
   * @brief     Destructor of TensorPool
   */
  ~TensorPool() = default;

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
   * @brief     Get execution order for the given tensor
   *
   * @return The execution order of the tensor
   */
  const std::vector<unsigned int> &getExecutionOrder(const std::string &name);

  /**
   * @brief Get the maximum real memory requirement
   *
   * @return The real memory requirement with this strategy in bytes
   */
  size_t size() { return mem_pool->size(); }

  /**
   * @brief Get the minimum theoretical memory requirement
   *
   * @return The theoretical memory requirement with this strategy in bytes
   */
  size_t minMemoryRequirement() { return mem_pool->minMemoryRequirement(); }

  /**
   * @brief Is the tensor pool allocated
   *
   * @return true if the tensors are allocated, else false
   */
  bool isAllocated() const { return mem_pool->isAllocated(); }

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
  void fillPlaceholder(const std::string &name, const Tensor &t);

  /**
   * @brief request placeholder which will be not managed by this tensor pool
   * but will be managed externally
   *
   * @param name Name of the tensor
   * @param dim Tensor dimension
   * @return Tensor* ptr to the tensor
   *
   * @note returns empty tensor which must be filled by the caller before use.
   */
  Tensor *placeholder(const std::string &name, const TensorDim &dim);

  /**
   * @brief     create a new tensor with the given spec.
   *
   * @param name Name of this tensor.
   * @param dim Tensor dimension.
   * @param exec_order The execution orders for this tensor.
   * @param lifespan Lifespan of this tensor.
   * @param init Initializer of the tensor.
   * @param is_weight_grad Identification of weight gradient
   *
   * @return ptr to the created tensor
   *
   * @note returns empty tensor which will be filled when allocate is called.
   * @note we assume that the caller checks if the exec_order and lifespan are
   * compatible.
   */
  Tensor *request(const std::string &name, const TensorDim &dim,
                  const std::vector<unsigned int> &exec_order,
                  TensorLifespan lifespan,
                  const Tensor::Initializer &init = Tensor::Initializer::NONE,
                  bool is_weight_grad = false);

  /**
   * @brief     Request tensor which is a view of already requested with the
   * given spec
   *
   * @param name Name of this tensor
   * @param reference Name of the reference tensor
   * @param dim Tensor dimensions
   * @param exec_order The execution orders for this tensors
   * @param lifespan Lifespan of this tensor
   * @param offset offset from the reference
   *
   * @return ptr to a tensor which is sharing the same data with
   * reference.
   *
   * @note returns a view tensor which will be filled when the source tensor is
   * allocated.
   * @note we assume that the caller checks if the exec_order and lifespan are
   * compatible.
   *
   */
  Tensor *view(const std::string &name, const std::string &reference,
               const TensorDim &dim,
               const std::vector<unsigned int> &exec_order,
               TensorLifespan lifespan, const unsigned int offset = 0);

  /**
   * @brief extend a tensor life as tensor is being shared.
   *
   * @param name name of the tensor to extend
   * @param dim dimension of the tensor
   * @param exec_order exec_order to extend
   * @param lifespan extended life span
   * @return Tensor* Tensor* the exact tensor which is being extended.
   * @note we assume that the caller checks if the exec_order and lifespan are
   * compatible.
   */
  Tensor *extend(const std::string &name, const TensorDim &dim,
                 const std::vector<unsigned int> &exec_order,
                 TensorLifespan lifespan);

  /**
   * @brief create a new tensor if tensor does not exist else return the tensor
   * while extending the tensor's life according to the given arguments.
   * @note Created (or extended) tensor is considered identical and managed. It
   * is invalid to create a tensor with lifespan::UNMANAGED or dimension and
   * initializer is different upon extension.
   *
   * @param name Name of the tensor
   * @param dim dimension
   * @param exec_order exec order
   * @param lifespan tensor life span
   * @param init tensor initializer
   * @return Tensor* ptr to either to the existing tensor or newly created
   * tensor
   */
  Tensor *
  requestOrExtend(const std::string &name, const TensorDim &dim,
                  const std::vector<unsigned int> &exec_order,
                  TensorLifespan lifespan,
                  const Tensor::Initializer &init = Tensor::Initializer::NONE);

  /**
   * @brief reidentify the source of already created tensor (or view).
   * @note if @a dest tensor is a view of another tensor, the old source tensor
   * of the view will become a view of @a new_src.
   *
   * @throws std::invalid_argument 1. if the data size required from the
   * original source tensor is bigger than the new_src + offset. 2. if new_src
   * is a view. Second restriction can be removed, if this is considered as a
   * safe behavior.
   *
   * @param dest identifier for the dest tensor
   * @param new_src identifier for the new source tensor
   * @param offset offset
   */
  void reidentifySource(const std::string &dest, const std::string &new_src,
                        unsigned int offset);

  /**
   * @brief flush cache data
   *
   */
  void flushCache();

  /**
   * @brief flush cache data except order
   *
   * @param order except execution order
   *
   */
  void flushCacheExcept(unsigned int order);

private:
  /**
   * @brief Source tensor detailed specification
   *
   */
  struct SourceDetails {
    unsigned int token;                   /**< memory token */
    TensorLifespan lifespan;              /**< life span of the tensor */
    std::vector<unsigned int> exec_order; /**< exec order */
    std::vector<unsigned int>
      dependents; /**< list of dependents to the source */
  };

  /**
   * @brief Dependent tensor detaild specification
   *
   */
  struct DependentDetails {
    unsigned int parent_idx; /**< index to the parent */
    unsigned int offset;     /**< elementwise offset */
  };

  /**
   * @brief Spec for storing each request of tensor from tensor pool
   * @todo move tensor initialization from tensor class to RequestSpec
   */
  struct RequestSpec {
    bool is_weight_grad;            /**< identification of weight gradient */
    std::unique_ptr<Tensor> tensor; /**< tensor object itself */
    std::variant<SourceDetails, DependentDetails>
      details; /**< additional information by its kind */
  };

  /**
   * @brief check if a tensor exist with the given identifier
   *
   * @param name name name to check
   * @retval true if exist
   * @retval false if do not exist
   */
  bool tensorExist(const std::string &name);

  /**
   * @brief Get the view of source Spec from the name
   *
   * @param name name to get source spec
   * @return RequestSpec spec
   */
  RequestSpec &getSourceSpec(const std::string &name);

  /**
   * @brief     Expand the lifespan of the tensor with the given name
   *
   * @param name The name of the tensor
   * @param exec_order The execution orders
   * @param lifespan The lifespan to be expanded to
   * @return source spec for the name
   */
  RequestSpec &expandLifespan(const std::string &name,
                              const std::vector<unsigned int> &exec_order,
                              TensorLifespan lifespan);

  /**
   * @brief expand life span with execution time
   *
   * @param spec specification
   * @param exec_order exec order
   * @param lifespan life span
   */
  void expandLifespan(RequestSpec &spec,
                      const std::vector<unsigned int> &exec_order,
                      TensorLifespan lifespan);

  /**
   * @brief sync dependent tensors from updated source tensor
   * @note syncing starting from dependents of dependents is invalid and will
   * throw.
   *
   * @param spec spec with source details to refer to.
   */
  void syncDependents(const RequestSpec &spec);

  /**
   * @brief register a spec after creation
   *
   * @param spec spec to register
   */
  Tensor *registerRequestSpec(RequestSpec &&spec);

  /**
   * note: unordered_map is not directly used for pool to ensure initialization
   * of weights
   */
  std::vector<RequestSpec> pool; /**< list of requested tensors */
  std::unordered_map<std::string, unsigned int>
    name_map;                           /**< indexing of requested tensors */
  std::unique_ptr<MemoryPool> mem_pool; /**< memory pool for the tensors */

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
