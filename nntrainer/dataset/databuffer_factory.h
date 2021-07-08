// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   databuffer_factory.h
 * @date   19 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the layer factory.
 */

#ifndef __DATABUFFER_FACTORY_H__
#define __DATABUFFER_FACTORY_H__
#ifdef __cplusplus

#include <databuffer.h>

namespace nntrainer {

/**
 * @brief Factory creator with constructor
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type);

/**
 * @brief Factory creator with constructor for databuffer with files
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type,
                                             const char *train_file,
                                             const char *valid_file = nullptr,
                                             const char *test_file = nullptr);

/**
 * @brief Factory creator with constructor for databuffer with callbacks
 */
std::unique_ptr<DataBuffer> createDataBuffer(DatasetType type, datagen_cb train,
                                             datagen_cb valid = nullptr,
                                             datagen_cb test = nullptr);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __DATABUFFER_FACTORY_H__ */
