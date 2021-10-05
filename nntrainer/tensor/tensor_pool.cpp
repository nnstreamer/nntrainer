// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tensor_pool.cpp
 * @date   19 Aug 2020
 * @brief  This is TensorPool for all requested tensors
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 * @todo   add checks for request/updates that finalize is not done
 * @todo   check before allocate that finalize is done
 */

#include <memory_pool.h>
#include <nntrainer_log.h>
#include <tensor.h>
#include <tensor_pool.h>
#include <tensor_wrap_specs.h>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief     Request tensor with the given spec
 *
 * @note returns empty tensor which will be filled when allocate is called.
 * @note we assume that the caller checks if the exec_order and lifespan are
 * compatible.
 */
Tensor *TensorPool::requestTensor(const TensorDim &dim,
                                  const std::vector<unsigned int> &exec_order,
                                  TensorLifespan lifespan,
                                  const std::string &name,
                                  const Tensor::Initializer &init) {
  if (name_map.find(name) != name_map.end())
    throw std::invalid_argument("Cannot request tensor with same name");

  if (dim.getDataLen() == 0)
    throw std::invalid_argument("Cannot request tensor with size 0");

  if (name.empty())
    throw std::invalid_argument("Cannot request tensor with empty name");

  pool.push_back({std::make_unique<Tensor>(dim, false, init, name), exec_order,
                  lifespan, 0, false});
  name_map[name] = pool.size() - 1;

  return pool.back().tensor.get();
}

/**
 * @brief     Request tensor with the given spec
 *
 * @note returns empty tensor which will be filled when allocate is called.
 * @note we assume that the caller checks if the exec_order and lifespan are
 * compatible.
 */
Tensor *
TensorPool::requestExternallyAllocateTensor(const TensorDim &dim,
                                            const std::string &name,
                                            const Tensor::Initializer &init) {
  return requestTensor(dim, {}, TensorLifespan::ZERO_LIFESPAN, name, init);
}

/**
 * @brief     Request tensor which has been already requested with the given
 * spec
 *
 * @note returns empty tensor which will be filled when allocate is called.
 * @note we assume that the caller checks if the exec_order and lifespan are
 * compatible.
 */
Tensor *TensorPool::requestPrerequestedTensor(
  const TensorDim &dim, const std::vector<unsigned int> &exec_order,
  TensorLifespan lifespan, const std::string &name,
  const std::string &shared_name, const Tensor::Initializer &init) {

  auto &spec = getSourceSpec(shared_name);
  if (spec.tensor->getDim().getDataLen() != dim.getDataLen())
    throw std::invalid_argument("Request tensor dimension mismatch");

  if (init != Tensor::Initializer::NONE &&
      spec.tensor->getInitializer() != init)
    throw std::invalid_argument("Request tensor initialization mismatch");

  /**
   * cannot expand lifespan of zero lifespan tensor
   * it works for externally allocated tensors as well
   */
  if (spec.lifespan != TensorLifespan::ZERO_LIFESPAN) {
    spec.exec_order.insert(spec.exec_order.end(), exec_order.begin(),
                           exec_order.end());
    spec.lifespan = enum_class_or<TensorLifespan>(spec.lifespan, lifespan);
  }

  /** @note requestTensor invalidates spec reference */
  /// maybe bug: we should never access exec_order, lifespan of ret, we should
  /// only access pool[parent_spec_idx]
  Tensor *ret = requestTensor(dim, exec_order, lifespan, name, init);
  pool.back().token = name_map[shared_name];
  pool.back().dependent = true;

  return ret;
}

/**
 * @brief finalize the requested tensors
 *
 * @details finalize the requested tensors, request memory for them and plan
 * layout for their allocations.
 */
void TensorPool::finalize(const MemoryPlanner &planner,
                          unsigned int start_order, unsigned int end_order) {
  mem_pool.clear();
  unsigned int bytes_requested = 0;
  for (auto &spec : pool) {
    /** do not include dependent tensors in planning layout */
    if (spec.dependent || spec.exec_order.empty() ||
        spec.lifespan == TensorLifespan::ZERO_LIFESPAN)
      continue;

    spec.token = 0;

    /** 1. create the validity ranges for the all the requested tensors */
    unsigned int validity_start =
      *std::min_element(spec.exec_order.begin(), spec.exec_order.end());
    unsigned int validity_end =
      *std::max_element(spec.exec_order.begin(), spec.exec_order.end());

    /**
     * use lifespan to update the validity.
     * if the validity is long term, the tensor must stay valid for the
     * complete duration.
     */
    if (isTensorLongTerm(spec.lifespan)) {
      validity_start = start_order;
      validity_end = end_order;
    }

    /** 2. for each tensor request if it is in the provided range */
    if (validity_end < start_order || validity_start > end_order)
      continue;
    validity_start = std::max(validity_start, start_order);
    validity_end = std::min(validity_end, end_order);

    /**
     * 3. requestMemory for all the tensors and set their tokens
     * @note +1 is to make the validity_end exlusive in the interval range
     */
    spec.token = mem_pool.requestMemory(spec.tensor->bytes(), validity_start,
                                        validity_end + 1);
#ifdef DEBUG
    if (spec.token == 0)
      throw std::runtime_error("Received invalid token from memory pool");
#endif

    bytes_requested += spec.tensor->bytes();
  }

  /** 4. finalizeLayout for the memory pool. */
  if (bytes_requested > 0) {
    double efficiency = mem_pool.planLayout(planner);
    ml_logd("Memory layout efficiency = %lf", efficiency);
  }
}

/**
 * @brief Set the batch size for the inputs/outputs of the layers
 */
void TensorPool::setBatchSize(const std::string &name, unsigned int batch) {
  if (name_map.find(name) == name_map.end())
    throw std::invalid_argument("Requested tensor not found");

  pool[name_map[name]].tensor->updateBatch(batch);
}

/**
 * @brief Allocate memory for all the managed tensors
 */
void TensorPool::allocate() {
  mem_pool.allocate();

  /** set the pointers using the token for all the tensors */
  for (auto &spec : pool) {
    /** get data for the tensors which were requested */
    if (!spec.dependent && spec.token > 0) {
      spec.tensor->setData(mem_pool.getMemory(spec.token));
      spec.tensor->initialize();
    } else if (spec.dependent) {
      spec.tensor->setData(pool[spec.token].tensor->getData());
      /** initialize is intentionally skipped here */
    }
  }
}

/**
 * @brief Deallocate memory for all the managed tensors
 */
void TensorPool::deallocate() {
  mem_pool.deallocate();

  /** nullify the data pointers for the tensors */
  for (auto &spec : pool) {
    spec.tensor->setData(nullptr);
  }
}

/**
 * @brief     Expand the lifespan of the tensor with the given name
 *
 */
void TensorPool::expand_lifespan(const std::string &name,
                                 TensorLifespan lifespan) {
  if (name_map.find(name) == name_map.end())
    throw std::invalid_argument("Requested tensor not found");

  int parent_spec_idx = name_map[name];
  while (pool[parent_spec_idx].dependent == true)
    parent_spec_idx = pool[parent_spec_idx].token;

  auto &spec = pool[parent_spec_idx];

  if (spec.lifespan != TensorLifespan::ZERO_LIFESPAN)
    throw std::invalid_argument("Cannot extend tensor lifespan from ZERO");

  spec.lifespan = enum_class_or<TensorLifespan>(spec.lifespan, lifespan);
}

/**
 * @brief     Expand the execution order of the tensor with the given name
 *
 */
void TensorPool::expand_lifespan(const std::string &name,
                                 const std::vector<unsigned int> &exec_order) {
  if (name_map.find(name) == name_map.end())
    throw std::invalid_argument("Requested tensor not found");

  int parent_spec_idx = name_map[name];
  while (pool[parent_spec_idx].dependent == true)
    parent_spec_idx = pool[parent_spec_idx].token;

  auto &spec = pool[parent_spec_idx];
  spec.exec_order.insert(spec.exec_order.end(), exec_order.begin(),
                         exec_order.end());
}

TensorPool::requestSpec &TensorPool::getSourceSpec(const std::string &name) {
  unsigned parent_spec_idx;
  try {
    parent_spec_idx = name_map.at(name);
  } catch (...) {
    throw std::invalid_argument("finding spec idx failed, name: " + name);
  }
  while (pool[parent_spec_idx].dependent == true)
    parent_spec_idx = pool[parent_spec_idx].token;

  return pool.at(parent_spec_idx);
}

/**
 * @brief     Check if the lifespan leads to long term valitidy
 *
 */
bool TensorPool::isTensorLongTerm(const TensorLifespan &lifespan) {
  switch (lifespan) {
  case TensorLifespan::EPOCH_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::MAX_LIFESPAN:
    return true;
  case TensorLifespan::FORWARD_FUNC_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::BACKWARD_FUNC_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::ITERATION_LIFESPAN:
    [[fallthrough]];
  case TensorLifespan::ZERO_LIFESPAN:
    [[fallthrough]];
  default:
    return false;
  }
}

} // namespace nntrainer
