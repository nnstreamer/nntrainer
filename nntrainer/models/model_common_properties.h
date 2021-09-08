// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   model_common_properties.h
 * @date   27 Aug 2021
 * @brief  This file contains common properties for model
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __MODEL_COMMON_PROPERTIES_H__
#define __MODEL_COMMON_PROPERTIES_H__

#include <base_properties.h>

#ifdef __cplusplus
namespace nntrainer::props {

/**
 * @brief model epoch property
 *
 */
class Epochs : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "epochs"; /**< unique key to access */
  using prop_tag = uint_prop_tag;              /**< property type */
  /**
   * @brief Construct a new Epochs object
   *
   * @param value value to set
   */
  Epochs(unsigned int value = 1);
};

/**
 * @brief model loss property (deprecated)
 *
 */
class LossType : public Property<std::string> {
public:
  static constexpr const char *key = "loss"; /**< unique key to access */
  using prop_tag = str_prop_tag;             /**< property type */

  /**
   * @brief check if valid
   *
   * @param value value to check
   * @return bool true if valid
   */
  bool isValid(const std::string &value) const override;
};

/**
 * @brief model save path property
 *
 */
class SavePath : public Property<std::string> {
public:
  static constexpr const char *key = "save_path"; /**< unique key to access */
  using prop_tag = str_prop_tag;                  /**< property type */
};

/**
 * @brief model save path property
 *
 */
class SaveBestPath : public Property<std::string> {
public:
  static constexpr const char *key =
    "save_best_path";            /**< unique key to access */
  using prop_tag = str_prop_tag; /**< property type */
};

/**
 * @brief model batch size property
 *
 */
class TrainingBatchSize : public PositiveIntegerProperty {
public:
  static constexpr const char *key = "batch_size"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                  /**< property type */

  /**
   * @brief Construct a new Batch Size object
   *
   * @param value value to set, defaults to 1
   */
  TrainingBatchSize(unsigned int value = 1);
};

/**
 * @brief model continue property
 *
 */
class ContinueTrain : public Property<bool> {
public:
  static constexpr const char *key =
    "continue_train";             /**< unique key to access */
  using prop_tag = bool_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to false
   */
  ContinueTrain(bool value = false);
};

} // namespace nntrainer::props

#endif

#endif // __MODEL_COMMON_PROPERTIES_H__
