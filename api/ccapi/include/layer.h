// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	layer.h
 * @date	14 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is layers interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_LAYER_H__
#define __ML_TRAIN_LAYER_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <string>
#include <vector>

#include <nntrainer-api-common.h>

namespace ml {
namespace train {

/**
 * @brief     Enumeration of layer type
 */
enum class LayerType {
  LAYER_IN = ML_TRAIN_LAYER_TYPE_INPUT, /** Input Layer type */
  LAYER_FC = ML_TRAIN_LAYER_TYPE_FC,    /** Fully Connected Layer type */
  LAYER_BN,                             /** Batch Normalization Layer type */
  LAYER_CONV2D,                         /** Convolution 2D Layer type */
  LAYER_POOLING2D,                      /** Pooling 2D Layer type */
  LAYER_FLATTEN,                        /** Flatten Layer type */
  LAYER_ACTIVATION,                     /** Activation Layer type */
  LAYER_ADDITION,                       /** Addition Layer type */
  LAYER_LOSS,                           /** Loss Layer type */
  LAYER_UNKNOWN = ML_TRAIN_LAYER_TYPE_UNKNOWN /** Unknown */
};

/**
 * @class   Layer Base class for layers
 * @brief   Base class for all layers
 */
class Layer {
public:
  /**
   * @brief     Destructor of Layer Class
   */
  virtual ~Layer() = default;

  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(std::vector<std::string> values) = 0;

  /**
   * @brief     Property Enumeration
   *            0. input shape : string
   *            1. bias zero : bool
   *            2. normalization : bool
   *            3. standardization : bool
   *            4. activation : string (type)
   *            5. epsilon : float
   *            6. weight_regularizer : string (type)
   *            7. weight_regularizer_constant : float
   *            8. unit : int
   *            9. weight_initializer : string (type)
   *            10. filter_size : int
   *            11. kernel_size : ( n , m )
   *            12. stride : ( n, m )
   *            13. padding : ( n, m )
   *            14. pool_size : ( n,m )
   *            15. pooling : max, average, global_max, global_average
   *            16. flatten : bool
   *            17. name : string (type)
   *            18. num_inputs : unsigned int (minimum 1)
   *            19. num_outputs : unsigned int (minimum 1)
   *            20. momentum : float,
   *            21. moving_mean_initializer : string (type),
   *            22. moving_variance_initializer : string (type),
   *            23. gamma_initializer : string (type),
   *            24. beta_initializer" : string (type)
   */
  enum class PropertyType {
    input_shape = 0,
    normalization = 1,
    standardization = 2,
    activation = 3,
    epsilon = 4,
    weight_regularizer = 5,
    weight_regularizer_constant = 6,
    unit = 7,
    weight_initializer = 8,
    bias_initializer = 9,
    filters = 10,
    kernel_size = 11,
    stride = 12,
    padding = 13,
    pool_size = 14,
    pooling = 15,
    flatten = 16,
    name = 17,
    num_inputs = 18,
    num_outputs = 19,
    momentum = 20,
    moving_mean_initializer = 21,
    moving_variance_initializer = 22,
    gamma_initializer = 23,
    beta_initializer = 24,
    unknown
  };

  /**
   * @brief setProperty by PropertyType
   * @note By passing empty string, this can validate if @a type is valid
   * @param[in] type property type to be passed
   * @param[in] value value to be passed, if empty string is passed, do nothing
   * but throws error when @a type is invalid
   * @exception exception::not_supported     when property type is not valid for
   * the particular layer
   * @exception std::invalid_argument invalid argument
   * @note A layer need not support all the properties from PropertyType, but
   * the supported properties will be a subset of PropertyType.
   */
  virtual void setProperty(const PropertyType type,
                           const std::string &value = "") = 0;

  /**
   * @brief     check hyper parameter for the layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int checkValidation() = 0;

  /**
   * @brief  get the loss value added by this layer
   * @retval loss value
   */
  virtual float getLoss() = 0;

  /**
   * @brief     set trainable for this layer
   * @param[in] train to enable/disable train
   */
  virtual void setTrainable(bool train) = 0;

  /**
   * @brief     get if the output of this layer must be flatten
   * @retval    flatten value
   */
  virtual bool getFlatten() = 0;

  /**
   * @brief     Get name of the layer
   */
  virtual std::string getName() noexcept = 0;

  /**
   * @brief Preset modes for printing summary for the layer
   */
  enum class PrintPreset {
    PRINT_NONE = 0,     /**< Print nothing */
    PRINT_SUMMARY,      /**< Print preset including summary information */
    PRINT_SUMMARY_META, /**< Print summary preset that includes meta information
                         */
    PRINT_ALL           /**< Print everything possible */
  };

  /**
   * @brief print using PrintPreset
   *
   * @param out oustream
   * @param preset preset to be used
   */
  virtual void printPreset(std::ostream &out,
                           PrintPreset preset = PrintPreset::PRINT_SUMMARY) = 0;
};

/**
 * @brief Factory creator with constructor for layer
 */
std::unique_ptr<Layer>
createLayer(LayerType type, const std::vector<std::string> &properties = {});

} // namespace train
} // namespace ml

#else
#error "CPP versions c++14 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_LAYER_H__
