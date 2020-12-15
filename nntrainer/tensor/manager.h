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
 * @bug	   No known bugs except for NYI items
 *
 */

#ifndef __MANAGER_H__
#define __MANAGER_H__
#ifdef __cplusplus

#include <functional>
#include <vector>

#include <weight.h>

namespace nntrainer {

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
          bool use_shared_memory_ = true);

  /// @todo copy ctor / assignment ops but leave move ctor / assignment
  Manager(const Manager &) = default;

  Manager &operator=(const Manager &) = default;

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
   * @brief Get the File descriptor.
   * Will return -1 except for android
   * @todo make this available for other platforms
   *
   * @return -1 if not applicable, else file descriptor
   */
  int getFd() noexcept { return fd; }

  /**
   * @brief Allocate and initialize the weight variable
   */
  void initialize();

  /**
   * @brief Reset the manager state
   */
  void reset() {
    weights.clear();
    max_grad_size = 0;
    total_weight_size = 0;
    total_grad_size = 0;
    releaseSharedMemory();
  }

private:
  /**
   * @brief initialize shared memory, buf_size is set here
   *
   * @param size Byte size
   */
  void initializeSharedMemory(size_t size);

  /**
   * @brief release shared memory, if use_sha
   *
   */
  void releaseSharedMemory() noexcept;

  // TODO: ensure that names of these weights are unique
  /**< Weights all the layer in the model to be managed */
  std::vector<std::vector<std::reference_wrapper<Weight>>> weights;

  size_t total_weight_size; /**< total weight size */
  size_t total_grad_size;   /**< total weight size */
  size_t max_grad_size;     /**< max trainable weight required by a layer */

  bool enable_gradient_memory_opt; /**< share memory among all the gradients */

  /**< shared memory related */
  bool use_shared_memory; /**< uses shared memory object which is owned by
                             manager */
  int fd;                 /**< fd to access the shared_memory  */
  float *buf;             /**< buffer object when use_shared_memory */
  size_t buf_size;        /**< buffer size */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MANAGER_H__ */
