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
                                             const char *train_file,
                                             const char *valid_file,
                                             const char *test_file) {
  if (type != DatasetType::FILE)
    throw std::invalid_argument(
      "Cannot create dataset with files with the given dataset type");

  std::unique_ptr<DataBuffer> dataset = createDataBuffer(type);

  NNTR_THROW_IF(train_file == nullptr ||
                  dataset->setDataFile(DatasetDataUsageType::DATA_TRAIN,
                                       train_file) != ML_ERROR_NONE,
                std::invalid_argument)
    << "invalid train file, path: " << (train_file ? train_file : "null");

  if (valid_file) {
    NNTR_THROW_IF(dataset->setDataFile(DatasetDataUsageType::DATA_VAL,
                                       valid_file) != ML_ERROR_NONE,
                  std::invalid_argument)
      << "invalid valid file, path: " << (valid_file ? valid_file : "null");
  }

  if (test_file) {
    NNTR_THROW_IF(dataset->setDataFile(DatasetDataUsageType::DATA_TEST,
                                       test_file) != ML_ERROR_NONE,
                  std::invalid_argument)
      << "invalid test file, path: " << (test_file ? test_file : "null");
  }

  return dataset;
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type, datagen_cb train,
                                             datagen_cb valid,
                                             datagen_cb test) {
  if (type != DatasetType::GENERATOR)
    throw std::invalid_argument("Cannot create dataset with generator "
                                "callbacks with the given dataset type");

  std::unique_ptr<DataBuffer> dataset = createDataBuffer(type);

  if (dataset->setGeneratorFunc(DatasetDataUsageType::DATA_TRAIN, train) !=
      ML_ERROR_NONE)
    throw std::invalid_argument("Invalid train data generator");

  if (valid && dataset->setGeneratorFunc(DatasetDataUsageType::DATA_VAL,
                                         valid) != ML_ERROR_NONE)
    throw std::invalid_argument("Invalid valid data generator");

  if (test && dataset->setGeneratorFunc(DatasetDataUsageType::DATA_TEST,
                                        test) != ML_ERROR_NONE)
    throw std::invalid_argument("Invalid test data generator");

  return dataset;
}

} // namespace nntrainer
