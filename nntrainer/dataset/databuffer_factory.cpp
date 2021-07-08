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

#include <databuffer_file.h>
#include <databuffer_func.h>
#include <nntrainer_error.h>

namespace nntrainer {

/**
 * @brief Factory creator with constructor
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type) {
  switch (type) {
  case DatasetType::GENERATOR:
    return std::make_unique<DataBufferFromCallback>();
  case DatasetType::FILE:
    return std::make_unique<DataBufferFromDataFile>();
  case DatasetType::UNKNOWN:
    /** fallthrough intended */
  default:
    throw std::invalid_argument("Unknown type for the dataset");
  }
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type,
                                             const char *file) {
  if (type != DatasetType::FILE)
    throw std::invalid_argument(
      "Cannot create dataset with files with the given dataset type");

  std::unique_ptr<DataBuffer> dataset = createDataBuffer(type);

  NNTR_THROW_IF(file == nullptr ||
                  dataset->setDataFile(DatasetDataUsageType::DATA_TRAIN,
                                       file) != ML_ERROR_NONE,
                std::invalid_argument)
    << "invalid train file, path: " << (file ? file : "null");

  return dataset;
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type, datagen_cb cb,
                                             void *user_data) {
  if (type != DatasetType::GENERATOR)
    throw std::invalid_argument("Cannot create dataset with generator "
                                "callbacks with the given dataset type");

  std::unique_ptr<DataBuffer> dataset = createDataBuffer(type);

  if (dataset->setGeneratorFunc(DatasetDataUsageType::DATA_TRAIN, cb,
                                user_data) != ML_ERROR_NONE)
    throw std::invalid_argument("Invalid train data generator");

  return dataset;
}

} // namespace nntrainer
