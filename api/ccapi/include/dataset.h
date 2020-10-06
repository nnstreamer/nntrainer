// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	dataset.h
 * @date	14 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is dataset interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_DATASET_H__
#define __ML_TRAIN_DATASET_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <nntrainer-api-common.h>

namespace ml {
namespace train {

/**
 * @brief   Dataset generator callback type declaration
 */
typedef std::function<std::remove_pointer<ml_train_datagen_cb>::type>
  datagen_cb;

/**
 * @brief     Enumeration for dataset type
 */
enum class DatasetType {
  GENERATOR, /** Dataset with generators */
  FILE,      /** Dataset with files */
  UNKNOWN    /** Unknown dataset type */
};

/**
 * @brief     Enumeration of buffer type
 */
enum class BufferType {
  BUF_TRAIN,  /** BUF_TRAIN ( Buffer for training ) */
  BUF_VAL,    /** BUF_VAL ( Buffer for validation ) */
  BUF_TEST,   /** BUF_TEST ( Buffer for test ) */
  BUF_UNKNOWN /** BUF_UNKNOWN ( unknown ) */
};

/**
 * @class   Dataset for class for input data
 * @brief   Dataset for read and manage data
 */
class Dataset {
public:
  /**
   * @brief     Destructor
   */
  virtual ~Dataset() = default;

  /**
   * @brief     set function pointer for each type
   * @param[in] type Buffer Type
   * @param[in] call back function pointer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setFunc(BufferType type, datagen_cb func) = 0;

  /**
   * @brief     set property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(std::vector<std::string> values) = 0;
};

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset>
createDataset(DatasetType type,
              const std::vector<std::string> &properties = {});

} // namespace train
} // namespace ml

#else
#error "CPP versions c++14 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_DATASET_H__
