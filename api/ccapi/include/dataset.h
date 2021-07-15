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
 * @brief     Enumeration of data mode type
 */
enum class DatasetModeType {
  MODE_TRAIN = ML_TRAIN_DATASET_MODE_TRAIN, /** data for training */
  MODE_VALID = ML_TRAIN_DATASET_MODE_VALID, /** data for validation */
  MODE_TEST = ML_TRAIN_DATASET_MODE_TEST,   /** data for test */
  MODE_UNKNOWN                              /** data not known */
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
   * @details   Properties (values) is in the format -
   *  { std::string property_name, std::string property_val, ...}
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

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
 * @brief Create a Dataset object with given arguements
 *
 * @param type dataset type
 * @param properties property representations
 * @return std::unique_ptr<Dataset> created dataset
 */
std::unique_ptr<Dataset>
createDataset(DatasetType type,
              const std::vector<std::string> &properties = {});

/**
 * @brief Create a Dataset object
 *
 * @param type dataset type
 * @param path path to a file or folder
 * @param properties property representations
 * @return std::unique_ptr<Dataset> created dataset
 */
std::unique_ptr<Dataset>
createDataset(DatasetType type, const char *path,
              const std::vector<std::string> &properties = {});

/**
 * @brief Create a Dataset object
 *
 * @param type dataset type
 * @param cb callback
 * @param user_data user data
 * @param properties property representations
 * @return std::unique_ptr<Dataset> created dataset
 */
std::unique_ptr<Dataset>
createDataset(DatasetType type, datagen_cb cb, void *user_data = nullptr,
              const std::vector<std::string> &properties = {});
} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_DATASET_H__
