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
  case DatasetType::FILE:
    return std::make_unique<DataBufferFromDataFile>();
  case DatasetType::UNKNOWN:
    [[fallthrough]];
  default:
    throw std::invalid_argument("Unsupported constructor type for the dataset");
  }
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type,
                                             const char *file) {
  NNTR_THROW_IF(file == nullptr, std::invalid_argument)
    << "file shall not be null, use empty constructor instead";

  switch (type) {
  case DatasetType::FILE:
    return std::make_unique<DataBufferFromDataFile>(file);
  case DatasetType::UNKNOWN:
    [[fallthrough]];
  default:
    throw std::invalid_argument(
      "Unsupported constructor type for the dataset of type: " +
      static_cast<int>(type));
  };
}

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type, datagen_cb cb,
                                             void *user_data) {
  switch (type) {
  case DatasetType::GENERATOR:
    return std::make_unique<DataBufferFromCallback>(cb, user_data);
  case DatasetType::UNKNOWN:
    [[fallthrough]];
  default:
    throw std::invalid_argument(
      "Unsupported constructor type for the dataset of type: " +
      static_cast<int>(type));
  };
}

} // namespace nntrainer
