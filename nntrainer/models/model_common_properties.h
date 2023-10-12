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

/**
 * @brief model optimization property
 *
 */
class MemoryOptimization : public Property<bool> {
public:
  static constexpr const char *key =
    "memory_optimization";        /**< unique key to access */
  using prop_tag = bool_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to true
   */
  MemoryOptimization(bool value = true);
};

/**
 * @brief cache size property
 *
 */
class MemorySwap : public Property<bool> {
public:
  static constexpr const char *key = "memory_swap"; /**< unique key to access */
  using prop_tag = bool_prop_tag;                   /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to false
   */
  MemorySwap(bool value = false);
};

/**
 * @brief cache file path property
 *
 */
class MemorySwapPath : public Property<std::string> {
public:
  static constexpr const char *key =
    "memory_swap_path";          /**< unique key to access */
  using prop_tag = str_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to current directory
   */
  MemorySwapPath(const std::string &value = ".");
};

/**
 * @brief cache file path property
 *
 */
class MemorySwapLookahead : public Property<unsigned int> {
public:
  static constexpr const char *key =
    "memory_swap_lookahead";      /**< unique key to access */
  using prop_tag = uint_prop_tag; /**< property type */

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to current directory
   */
  MemorySwapLookahead(const unsigned int &value = 0);
};

/**
 * @brief     Enumeration of Data Type for model & layer
 */
struct ModelTensorDataTypeInfo {
  enum Enum { W4A16, W4A32, W8A16, W8A32, W16A16, W16A32, W32A16, W32A32 };
  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::W4A16,  Enum::W4A32,  Enum::W8A16,  Enum::W8A32,
    Enum::W16A16, Enum::W16A32, Enum::W32A16, Enum::W32A32};

  static constexpr const char *EnumStr[] = {
    "QINT4-FP16", "QINT4-FP32", "QINT8-FP16", "QINT8-FP32",
    "FP16-FP16",  "FP16-FP32",  "FP32-FP16",  "FP32-FP32"};
};

/**
 * @brief Activation Enumeration Information
 *
 */
class ModelTensorDataType final : public EnumProperty<ModelTensorDataTypeInfo> {
public:
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "model_tensor_type";

  /**
   * @brief Constructor
   *
   * @param value value to set, defaults to W32A32
   */
  ModelTensorDataType(ModelTensorDataTypeInfo::Enum value =
                        ModelTensorDataTypeInfo::Enum::W32A32);
};

} // namespace nntrainer::props

#endif

#endif // __MODEL_COMMON_PROPERTIES_H__
