// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   data_producers.h
 * @date   09 July 2021
 * @brief  This file contains data producer interface
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __DATA_PRODUCERS_H__
#define __DATA_PRODUCERS_H__

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
   * @brief Iteration represent a single batch which will be in a queue
   * @todo move this to data_buffer
   * @return std::get<0>(Iteration) denotes whether this is last iteration or
   * not, if true, std::get<1>(Iteration), std::get<2>(Iteration) will be
   * ignored
   * @return std::get<1>(Iteration) denotes inputs
   * @return std::get<2>(Iteration) denotes labels
   *
   */
  using Iteration = std::tuple<bool, std::vector<Tensor>, std::vector<Tensor>>;

  /**
   * @brief create an iteration
   * @todo rename this to BatchGenerator
   * @return Iteration iteration, if std::get<0>(retval) == true means end of
   * iteration, at the end of the iteration, it's responsibility of @a this to
   * shuffle.
   */
  using Generator = std::function<Iteration(void)>;

  /**
   * @brief Sample represents a view of single element which can be fed to the
   * model. It is the smallest unit to produce a data
   * @return std::get<0>(Sample) denotes inputs
   * @return std::get<1>(Sample) denotes labels
   */
  using Sample = std::tuple<std::vector<Tensor *>, std::vector<Tensor *>>;

  /**
   * @brief generator callable type which will fill a sample
   * @todo rename this to Generator.
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
  using Generator_sample =
    std::function<bool(unsigned int, /** index */
                       std::vector<Tensor *> & /** inputs */,
                       std::vector<Tensor *> & /** labels */)>;

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
   * @brief finalize the class with given properties
   * @todo remove this
   * @return Generator generator is a function that generates an iteration upon
   * call
   *
   */
  // [[deprecated("use finalize_sample instead")]]
  virtual Generator finalize(const std::vector<TensorDim> &input_dims,
                             const std::vector<TensorDim> &label_dims) = 0;

  /**
   * @brief finalize the class to return a immutable Generator.
   * @todo rename this to finalize.
   * @remark this function must assume that the batch dimension of each tensor
   * dimension is one. If actual dimension is not one, this function must ignore
   * the batch dimension and assume it to be one.
   * @param input_dims input dimensions.
   * @param label_dims label dimensions.
   * @param user_data user data to be used when finalize.
   * @return Generator generator is a function that generates a sample upon
   * call.
   */
  virtual Generator_sample
  finalize_sample(const std::vector<TensorDim> &input_dims,
                  const std::vector<TensorDim> &label_dims,
                  void *user_data = nullptr) {
    return Generator_sample();
  }

  /**
   * @brief get size of total dataset batch_size given input_dims, label_dims,
   * if size cannot be determined, this function must return
   * DataProducer::SIZE_UNDEFINED;
   *
   * @param input_dims input dimensions
   * @param label_dims label dimensions
   *
   * @return size calculated size
   */
  // [[deprecated("use size_sample instead")]]
  virtual unsigned int size(const std::vector<TensorDim> &input_dims,
                            const std::vector<TensorDim> &label_dims) const {
    return SIZE_UNDEFINED;
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
  virtual unsigned int
  size_sample(const std::vector<TensorDim> &input_dims,
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
#endif // __DATA_PRODUCERS_H__
