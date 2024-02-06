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

#include <common_properties.h>
#include <data_producer.h>
#include <dataset.h>

#include <memory>
#include <string>
#include <vector>

namespace nntrainer {

class Exporter;

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

  inline static const std::string type = "callback";

  /**
   * @copydoc DataProducer::getType()
   */
  const std::string getType() const override;

  /**
   * @copydoc DataProducer::setProperty(const std::vector<std::string>
   * &properties)
   */
  void setProperty(const std::vector<std::string> &properties) override;

  /**
   * @copydoc DataProducer::finalize(const std::vector<TensorDim>, const
   * std::vector<TensorDim>, void* user_data)
   */
  DataProducer::Generator finalize(const std::vector<TensorDim> &input_dims,
                                   const std::vector<TensorDim> &label_dims,
                                   void *user_data = nullptr) override;

  /**
   * @copydoc DataProducer::exportTo(Exporter &exporter,
   * ml::train::ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

private:
  datagen_cb cb;
  std::unique_ptr<props::PropsUserData> user_data_prop;
};

} // namespace nntrainer

#endif // __FUNC_DATA_PRODUCER_H__
