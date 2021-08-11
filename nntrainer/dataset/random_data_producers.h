// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   random_data_producers.h
 * @date   09 July 2021
 * @brief  This file contains various random data producers
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __RANDOM_DATA_PRODUCER_H__
#define __RANDOM_DATA_PRODUCER_H__

#include <data_producers.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace nntrainer {

class PropsMin;
class PropsMax;
class PropsNumSamples;

/**
 * @brief RandomDataProducer which generates a onehot vector as a label
 *
 */
class RandomDataOneHotProducer final : public DataProducer {
public:
  /**
   * @brief Construct a new Random Data One Hot Producer object
   *
   */
  RandomDataOneHotProducer();

  /**
   * @brief Destroy the Random Data One Hot Producer object
   *
   */
  ~RandomDataOneHotProducer();

  inline static const std::string type = "random_data_one_hot";

  /**
   * @copydoc DataProducer::getType()
   */
  const std::string getType() const override;

  /**
   * @copydoc DataProducer::isMultiThreadSafe()
   */
  bool isMultiThreadSafe() const override;

  /**
   * @copydoc DataProducer::size()
   */
  unsigned int size(const std::vector<TensorDim> &input_dims,
                    const std::vector<TensorDim> &label_dims) const override;

  /**
   * @copydoc DataProducer::setProeprty(const std::vector<std::string>
   * &properties)
   */
  void setProperty(const std::vector<std::string> &properties) override;

  /**
   * @copydoc DataProducer::finalize(const std::vector<TensorDim>, const
   * std::vector<TensorDim>)
   */
  DataProducer::Generator
  finalize(const std::vector<TensorDim> &input_dims,
           const std::vector<TensorDim> &label_dims) override;

  /**
   * @copydoc DataProducer::finalize_sample(const std::vector<TensorDim>, const
   * std::vector<TensorDim>, void *)
   */
  DataProducer::Generator_sample
  finalize_sample(const std::vector<TensorDim> &input_dims,
                  const std::vector<TensorDim> &label_dims,
                  void *user_data = nullptr) override;

  /**
   * @copydoc DataProducer::size_sample(const std::vector<TensorDim>, const
   * std::vector<TensorDim>)
   */
  unsigned int
  size_sample(const std::vector<TensorDim> &input_dims,
              const std::vector<TensorDim> &label_dims) const override;

private:
  using Props = std::tuple<PropsMin, PropsMax, PropsNumSamples>;
  std::unique_ptr<Props> rd_one_hot_props;
};

} // namespace nntrainer

#endif // __RANDOM_DATA_PRODUCER_H__
