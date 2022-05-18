// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   raw_file_data_producer.h
 * @date   12 July 2021
 * @brief  This file contains raw file data producers, reading from a file
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __RAW_FILE_DATA_PRODUCER_H__
#define __RAW_FILE_DATA_PRODUCER_H__

#include <data_producer.h>

#include <dataset.h>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace nntrainer {

namespace props {
class FilePath;
}

using datagen_cb = ml::train::datagen_cb;

/**
 * @brief RawFileDataProducer which contains a callback and returns back
 *
 */
class RawFileDataProducer final : public DataProducer {
public:
  inline static constexpr unsigned int pixel_size =
    sizeof(float); /**< @todo make this a configurable type */

  /**
   * @brief Construct a new Raw File Data Producer object
   *
   */
  RawFileDataProducer();

  /**
   * @brief Construct a new RawFileDataProducer object
   *
   */
  RawFileDataProducer(const std::string &path);

  /**
   * @brief Destroy the RawFileDataProducer object
   *
   */
  ~RawFileDataProducer();

  inline static const std::string type = "file";

  /**
   * @copydoc DataProducer::getType()
   */
  const std::string getType() const override;

  /**
   * @copydoc DataProducer::setProeprty(const std::vector<std::string>
   * &properties)
   */
  void setProperty(const std::vector<std::string> &properties) override;

  /**
   * @copydoc DataProducer::finalize(const std::vector<TensorDim>, const
   * std::vector<TensorDim>)
   */
  DataProducer::Generator finalize(const std::vector<TensorDim> &input_dims,
                                   const std::vector<TensorDim> &label_dims,
                                   void *user_data = nullptr) override;

  /**
   * @copydoc DataProducer::size(const std::vector<TensorDim>, const
   * std::vector<TensorDim>)
   */
  unsigned int size(const std::vector<TensorDim> &input_dims,
                    const std::vector<TensorDim> &label_dims) const override;

  /**
   * @copydoc DataProducer::exportTo(Exporter &exporter,
   * ml::train::ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

private:
  std::ifstream file;
  using PropTypes = std::tuple<props::FilePath>;
  std::unique_ptr<PropTypes> raw_file_props;
};

} // namespace nntrainer

#endif // __RAW_FILE_DATA_PRODUCER_H__
