// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dir_data_producers.h
 * @date   24 Feb 2023
 * @brief  This file contains dir data producers, reading from the files in
 * directory
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __DIR_DATA_PRODUCER_H__
#define __DIR_DATA_PRODUCER_H__

#include <data_producer.h>

#include <memory>
#include <random>
#include <string>
#include <vector>

namespace nntrainer {

namespace props {
class DirPath;
}

/**
 * @brief DirDataProducer which generates a onehot vector as a label
 *
 */
class DirDataProducer final : public DataProducer {
public:
  /**
   * @brief Construct a new Dir Data Producer object
   *
   */
  DirDataProducer();

  /**
   * @brief Construct a new Dir Data Producer object
   *
   */
  DirDataProducer(const std::string &dir_path);

  /**
   * @brief Destroy the Dir Data Producer object
   *
   */
  ~DirDataProducer();

  inline static const std::string type = "dir";

  /**
   * @copydoc DataProducer::getType()
   */
  const std::string getType() const override;

  /**
   * @copydoc DataProducer::isMultiThreadSafe()
   */
  bool isMultiThreadSafe() const override;

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
   * @copydoc DataProducer::finalize_sample(const std::vector<TensorDim>, const
   * std::vector<TensorDim>, void *)
   */
  unsigned int size(const std::vector<TensorDim> &input_dims,
                    const std::vector<TensorDim> &label_dims) const override;

private:
  using Props = std::tuple<props::DirPath>;
  std::unique_ptr<Props> dir_data_props;
  unsigned int num_class;
  size_t num_data_total;
  std::vector<std::pair<unsigned int, std::string>> data_list;
  std::vector<std::string> class_names;
};

} // namespace nntrainer

#endif // __DIR_DATA_PRODUCER_H__
