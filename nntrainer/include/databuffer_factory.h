// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	databuffer_factory.h
 * @date	19 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is the layer factory.
 */

#ifndef __DATABUFFER_FACTORY_H__
#define __DATABUFFER_FACTORY_H__
#ifdef __cplusplus

#include <databuffer.h>
#include <databuffer_file.h>
#include <databuffer_func.h>

namespace nntrainer {

/**
 * @brief Factory creator with constructor
 */
std::unique_ptr<DataBuffer> createDataBuffer(DataBufferType type) {
  switch (type) {
  case DataBufferType::GENERATOR:
    return std::make_unique<DataBufferFromCallback>();
  case DataBufferType::FILE:
    return std::make_unique<DataBufferFromDataFile>();
  case DataBufferType::UNKNOWN:
    /** fallthrough intended */
  default:
    throw std::invalid_argument("Unknown type for the dataset");
  }
}

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __DATABUFFER_FACTORY_H__ */
