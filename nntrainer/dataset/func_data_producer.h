// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   func_data_producer.h
 * @date   12 July 2021
 * @brief  This file contains various data producers from a callback
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __FUNC_DATA_PRODUCER_H__
#define __FUNC_DATA_PRODUCER_H__

#include <data_producers.h>

#include <dataset.h>

#include <memory>
#include <string>
#include <vector>

namespace nntrainer {

using datagen_cb = ml::train::datagen_cb;

/**
 * @brief FuncDataProducer which contains a callback and returns back
 *
 */
class FuncDataProducer final : public DataProducer {
public:
  /**
   * @brief Construct a new Func Data Producer object
   *
   * @param datagen_cb data callback
   * @param user_data_ user data
   */
  FuncDataProducer(datagen_cb datagen_cb, void *user_data_);

  /**
   * @brief Destroy the Func Data Producer object
   *
   */
  ~FuncDataProducer();

  inline static const std::string type = "batch_callback";

  /**
   * @copydoc DataProducer::getType()
   */
  const std::string getType() const override;

  /**
   * @copydoc DataProducer::setProeprty(const std::vector<std::string>
   * &properties)
   */
  virtual void setProperty(const std::vector<std::string> &properties) override;

  /**
   * @copydoc DataProducer::finalize(const std::vector<TensorDim>, const
   * std::vector<TensorDim>)
   */
  virtual DataProducer::Gernerator
  finalize(const std::vector<TensorDim> &input_dims,
           const std::vector<TensorDim> &label_dims) override;

private:
  datagen_cb cb;
  void *user_data;
};

} // namespace nntrainer

#endif // __FUNC_DATA_PRODUCER_H__
