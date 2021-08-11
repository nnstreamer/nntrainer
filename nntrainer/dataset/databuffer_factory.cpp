// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   databuffer_factory.cpp
 * @date   11 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the databuffer factory.
 */

#include <databuffer_factory.h>

#include <data_producer.h>
#include <func_data_producer.h>
#include <nntrainer_error.h>
#include <raw_file_data_producer.h>

namespace nntrainer {

/**
 * @brief Factory creator with constructor
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type) {

  std::unique_ptr<DataProducer> dp;
  switch (type) {
  case DatasetType::FILE:
    dp = std::make_unique<RawFileDataProducer>();
    break;
  case DatasetType::UNKNOWN:
    [[fallthrough]];
  default:
    throw std::invalid_argument("Unsupported constructor type for the dataset");
  }

  return std::make_unique<DataBuffer>(std::move(dp));
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type,
                                             const char *file) {
  NNTR_THROW_IF(file == nullptr, std::invalid_argument)
    << "file shall not be null, use empty constructor instead";

  std::unique_ptr<DataProducer> dp;
  switch (type) {
  case DatasetType::FILE:
    dp = std::make_unique<RawFileDataProducer>(file);
    break;
  case DatasetType::UNKNOWN:
    [[fallthrough]];
  default:
    throw std::invalid_argument(
      "Unsupported constructor type for the dataset of type: " +
      static_cast<int>(type));
  };

  return std::make_unique<DataBuffer>(std::move(dp));
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type, datagen_cb cb,
                                             void *user_data) {

  std::unique_ptr<DataProducer> dp;
  switch (type) {
  case DatasetType::GENERATOR:
    dp = std::make_unique<FuncDataProducer>(cb, user_data);
    break;
  case DatasetType::UNKNOWN:
    [[fallthrough]];
  default:
    throw std::invalid_argument(
      "Unsupported constructor type for the dataset of type: " +
      static_cast<int>(type));
  };

  return std::make_unique<DataBuffer>(std::move(dp));
}

} // namespace nntrainer
