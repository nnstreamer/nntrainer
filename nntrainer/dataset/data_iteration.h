// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   data_iteration.h
 * @date   11 Aug 2021
 * @brief  This file contains iteration and sample class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __DATA_SAMPLE_H__
#define __DATA_SAMPLE_H__

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include <tensor.h>
#include <tensor_dim.h>

namespace nntrainer {

class Sample;

/**
 * @brief Iteration class which owns the memory chunk for a single batch
 *
 */
class Iteration {

public:
  /**
   * @brief Construct a new Iteration object
   * @note the batch dimension must be the same for all given dimensions and the
   * there must be at least one input
   *
   * @param input_dims input dimension
   * @param label_dims label dimension
   */
  Iteration(const std::vector<ml::train::TensorDim> &input_dims,
            const std::vector<ml::train::TensorDim> &label_dims);

  Iteration(const Iteration &rhs) = delete;
  Iteration &operator=(const Iteration &rhs) = delete;
  Iteration(Iteration &&rhs) = default;
  Iteration &operator=(Iteration &&rhs) = default;

  /**
   * @brief get batch size of iteration
   *
   * @return unsigned int batch size
   */
  unsigned int batch();

  /**
   * @brief Get the Input Reference object
   *
   * @return std::vector<Tensor>& input
   */
  std::vector<Tensor> &getInputsRef() { return inputs; }

  /**
   * @brief Get the Input Reference object
   *
   * @return const std::vector<Tensor>& input
   */
  const std::vector<Tensor> &getInputsRef() const { return inputs; }

  /**
   * @brief Get the Label Reference object
   *
   * @return std::vector<Tensor>&  label
   */
  std::vector<Tensor> &getLabelsRef() { return labels; }

  /**
   * @brief Get the Label Reference object
   *
   * @return const std::vector<Tensor>&  label
   */
  const std::vector<Tensor> &getLabelsRef() const { return labels; }

  /**
   * @brief get sample iterator begin()
   *
   * @return std::vector<Sample>::iterator
   */
  std::vector<Sample>::iterator begin() { return samples.begin(); }

  /**
   * @brief get sample iterator end
   *
   * @return std::vector<Sample>::iterator
   */
  std::vector<Sample>::iterator end() { return end_iterator; }

  /**
   * @brief get sample iterator begin
   *
   * @return std::vector<Sample>::const_iterator
   */
  std::vector<Sample>::const_iterator begin() const { return samples.begin(); }

  /**
   * @brief get sample iterator end
   *
   * @return std::vector<Sample>::const_iterator
   */
  std::vector<Sample>::const_iterator end() const { return end_iterator; }

  /**
   * @brief set end of the sample which will be used to calculate the batch size
   * @note @a iteration must be non-inclusive
   *
   */
  void setEndSample(std::vector<Sample>::iterator sample_iterator);

  /**
   * @brief Set the End Sample to the original end
   *
   */
  void setEndSample();

private:
  std::vector<Tensor> inputs, labels;
  std::vector<Sample> samples;
  std::vector<Sample>::iterator end_iterator; /**< actual end iterator */
};

/**
 * @brief Sample class which views the memory for a single sample
 *
 */
class Sample {

public:
  /**
   * @brief Construct a new Sample object
   * @note the batch dimension will be ignored to make a single sample
   *
   * @param iter iteration objects
   * @param batch nth batch to create the sample
   */
  Sample(const Iteration &iter, unsigned int batch);

  Sample(const Sample &rhs) = delete;
  Sample &operator=(const Sample &rhs) = delete;
  Sample(Sample &&rhs) = default;
  Sample &operator=(Sample &&rhs) = default;

  /**
   * @brief Get the Input Reference object
   *
   * @return std::vector<Tensor>& input
   */
  std::vector<Tensor> &getInputsRef() { return inputs; }

  /**
   * @brief Get the Input Reference object
   *
   * @return const std::vector<Tensor>& input
   */
  const std::vector<Tensor> &getInputsRef() const { return inputs; }

  /**
   * @brief Get the Label Reference object
   *
   * @return std::vector<Tensor>&  label
   */
  std::vector<Tensor> &getLabelsRef() { return labels; }

  /**
   * @brief Get the Label Reference object
   *
   * @return const std::vector<Tensor>&  label
   */
  const std::vector<Tensor> &getLabelsRef() const { return labels; }

private:
  std::vector<Tensor> inputs, labels;
};

} // namespace nntrainer

#endif // __DATA_SAMPLE_H__
