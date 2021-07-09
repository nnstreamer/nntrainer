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
   * @return Iteration iteration, if std::get<0>(retval) == true means end of
   * iteration, at the end of the iteration, it's responsibility of @a this to
   * shuffle.
   */
  using Gernerator = std::function<Iteration(void)>;

  constexpr inline static unsigned long long SIZE_UNDEFINED =
    std::numeric_limits<unsigned long long>::max();

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
   * @return Generator generator is a function that generates an iteration upon
   * call
   *
   */
  virtual Gernerator finalize(const std::vector<TensorDim> &input_dims,
                              const std::vector<TensorDim> &label_dims) = 0;

  /**
   * @brief get size of total dataset given input_dims, label_dims, if size
   * cannot be determined, this function must return
   * DataProducer::SIZE_UNDEFINED;
   *
   * @param input_dims input dimensions
   * @param label_dims label dimensions
   *
   * @return size calculated size
   */
  virtual unsigned long long
  size(const std::vector<TensorDim> &input_dims,
       const std::vector<TensorDim> &label_dims) const {
    return SIZE_UNDEFINED;
  }
};
} // namespace nntrainer
#endif // __DATA_PRODUCERS_H__
