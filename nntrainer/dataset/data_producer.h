// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   data_producer.h
 * @date   09 July 2021
 * @brief  This file contains data producer interface
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __DATA_PRODUCER_H__
#define __DATA_PRODUCER_H__

#include <functional>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include <tensor.h>
#include <tensor_dim.h>
namespace nntrainer {

/**
 * @brief DataProducer interface used to abstract data provider
 *
 */
class DataProducer {
public:
  /**
   * @brief generator callable type which will fill a sample
   * @param[in] index current index with range of [0, size() - 1]. If
   * size() == SIZE_UNDEFINED, this parameter can be ignored
   * @param[out] inputs allocate tensor before expected to be filled by this
   * function
   * @param[out] labels allocate tensor before expected to be filled by this
   * function function.
   * @return bool true if this is the last sample, samples will NOT be ignored
   * and should be used, or passed at will of caller
   *
   */
  using Generator = std::function<bool(unsigned int, /** index */
                                       std::vector<Tensor> & /** inputs */,
                                       std::vector<Tensor> & /** labels */)>;

  constexpr inline static unsigned int SIZE_UNDEFINED =
    std::numeric_limits<unsigned int>::max();

  /**
   * @brief Destroy the Data Loader object
   *
   */
  virtual ~DataProducer() {}

  /**
   * @brief Get the producer type
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

  /**
   * @brief Set the Property object
   *
   * @param properties properties to set
   */
  virtual void setProperty(const std::vector<std::string> &properties) {
    if (!properties.empty()) {
      throw std::invalid_argument("There are unparsed properties");
    }
  }

  /**
   * @brief finalize the class to return an immutable Generator.
   * @remark this function must assume that the batch dimension of each tensor
   * dimension is one. If actual dimension is not one, this function must ignore
   * the batch dimension and assume it to be one.
   * @param input_dims input dimensions.
   * @param label_dims label dimensions.
   * @param user_data user data to be used when finalize.
   * @return Generator generator is a function that generates a sample upon
   * call.
   */
  virtual Generator finalize(const std::vector<TensorDim> &input_dims,
                             const std::vector<TensorDim> &label_dims,
                             void *user_data = nullptr) {
    return Generator();
  }

  /**
   * @brief get the number of samples inside the dataset, if size
   * cannot be determined, this function must return.
   * DataProducer::SIZE_UNDEFINED.
   * @remark this function must assume that the batch dimension of each tensor
   * dimension is one. If actual dimension is not one, this function must ignore
   * the batch dimension and assume it to be one
   * @param input_dims input dimensions
   * @param label_dims label dimensions
   *
   * @return size calculated size
   */
  virtual unsigned int size(const std::vector<TensorDim> &input_dims,
                            const std::vector<TensorDim> &label_dims) const {
    return SIZE_UNDEFINED;
  }

  /**
   * @brief denote if given producer is thread safe and can be parallelized.
   * @note if size() == SIZE_UNDEFIEND, thread safe shall be false
   *
   * @return bool true if thread safe.
   */
  virtual bool isMultiThreadSafe() const { return false; }
};
} // namespace nntrainer
#endif // __DATA_PRODUCER_H__
