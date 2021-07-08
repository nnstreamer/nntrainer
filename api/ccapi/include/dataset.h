
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   dataset.h
 * @date   14 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is dataset interface for c++ API
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
 * @brief     Enumeration of data type
 * @todo      deperecate data label
 */
enum class DatasetDataType {
  DATA_TRAIN = ML_TRAIN_DATASET_DATA_USAGE_TRAIN, /** data for training */
  DATA_VAL = ML_TRAIN_DATASET_DATA_USAGE_VALID,   /** data for validation */
  DATA_TEST = ML_TRAIN_DATASET_DATA_USAGE_TEST,   /** data for test */
  DATA_LABEL,                                     /** label names */
  DATA_UNKNOWN                                    /** data not known */
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
   * @brief     set property
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   Properties (values) is in the format -
   *  { std::string property_name, std::string property_val, ...}
   */
  virtual int setProperty(std::vector<std::string> values) = 0;

  /**
   * @brief     set property to allow setting non-string values such as
   * user_data for callbacks
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @note      this is a superset of the setProperty(std::vector<std::string>)
   * @details   Properties (values) is in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual int setProperty(std::vector<void *> values) = 0;
};

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset>
createDataset(DatasetType type,
              const std::vector<std::string> &properties = {});

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset> createDataset(DatasetType type, const char *train_file,
                                       const char *valid_file = nullptr,
                                       const char *test_file = nullptr);

/**
 * @brief Factory creator with constructor for dataset
 */
std::unique_ptr<Dataset> createDataset(DatasetType type, datagen_cb train,
                                       datagen_cb valid = nullptr,
                                       datagen_cb test = nullptr);

} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_DATASET_H__
