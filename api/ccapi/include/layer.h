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

#if __cplusplus < MIN_CPP_VERSION
#error "CPP versions c++14 or over are only supported"
#endif // __cpluscplus

#include <memory>
#include <string>
#include <vector>

#include <ml-api-common.h>
#include <nntrainer-api-common.h>

namespace ml {
namespace train {

/**
 * @brief     Enumeration of layer type
 */
enum LayerType {
  LAYER_IN = ML_TRAIN_LAYER_TYPE_INPUT, /** Input Layer type */
  LAYER_FC = ML_TRAIN_LAYER_TYPE_FC,    /** Fully Connected Layer type */
  LAYER_BN = ML_TRAIN_LAYER_TYPE_BN,    /** Batch Normalization Layer type */
  LAYER_CONV2D = ML_TRAIN_LAYER_TYPE_CONV2D, /** Convolution 2D Layer type */
  LAYER_POOLING2D = ML_TRAIN_LAYER_TYPE_POOLING2D, /** Pooling 2D Layer type */
  LAYER_FLATTEN = ML_TRAIN_LAYER_TYPE_FLATTEN,     /** Flatten Layer type */
  LAYER_ACTIVATION =
    ML_TRAIN_LAYER_TYPE_ACTIVATION,              /** Activation Layer type */
  LAYER_ADDITION = ML_TRAIN_LAYER_TYPE_ADDITION, /** Addition Layer type */
  LAYER_CONCAT = ML_TRAIN_LAYER_TYPE_CONCAT,     /** Concat Layer type */
  LAYER_MULTIOUT = ML_TRAIN_LAYER_TYPE_MULTIOUT, /** Multi Output Layer type */
  LAYER_LOSS,                                    /** Loss Layer type */
  LAYER_BACKBONE_NNSTREAMER,                  /** Backbone using NNStreamer */
  LAYER_BACKBONE_TFLITE,                      /** Backbone using TFLite */
  LAYER_EMBEDDING,                            /** Embedding Layer type */
  LAYER_RNN,                                  /** RNN Layer type */
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
   *            25. modelfile : model file for loading config for backbone layer
   *            26. input_layers : string (type)
   *            27. output_layers : string (type)
   *            28. trainable :
   *            29. flip_direction
   *            30. random_translate
   *            31. in_dim : int ( input dimension for embedding layer )
   *            32. out_dim : int ( output dimesion for embedding layer )
   *            33. in_length : int ( input length for embedding layer )
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
    modelfile = 25, /** model file for loading config for backbone layer */
    input_layers = 26,
    output_layers = 27,
    trainable = 28,
    flip_direction = 29,
    random_translate = 30,
    in_dim = 31,
    out_dim = 32,
    in_length = 33,
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
   * @brief Get the layer type
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

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
 * @brief Factory creator with constructor for layer type
 */
std::unique_ptr<Layer>
createLayer(const LayerType &type,
            const std::vector<std::string> &properties = {});

/**
 * @brief Factory creator with constructor for layer
 */
std::unique_ptr<Layer>
createLayer(const std::string &type,
            const std::vector<std::string> &properties = {});

namespace layer {

/**
 * @brief Helper function to create input layer
 */
inline std::unique_ptr<Layer>
Input(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_IN, properties);
}

/**
 * @brief Helper function to create fully connected layer
 */
inline std::unique_ptr<Layer>
FullyConnected(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_FC, properties);
}

/**
 * @brief Helper function to create batch normalization layer
 */
inline std::unique_ptr<Layer>
BatchNormalization(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_BN, properties);
}

/**
 * @brief Helper function to create convolution 2d layer
 */
inline std::unique_ptr<Layer>
Convolution2D(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CONV2D, properties);
}

/**
 * @brief Helper function to create pooling 2d layer
 */
inline std::unique_ptr<Layer>
Pooling2D(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_POOLING2D, properties);
}

/**
 * @brief Helper function to create flatten layer
 */
inline std::unique_ptr<Layer>
Flatten(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_FLATTEN, properties);
}

/**
 * @brief Helper function to create addition layer
 */
inline std::unique_ptr<Layer>
Addition(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_ADDITION, properties);
}

/**
 * @brief Helper function to create concat layer
 */
inline std::unique_ptr<Layer>
Concat(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_CONCAT, properties);
}

/**
 * @brief Helper function to create multi-out layer
 */
inline std::unique_ptr<Layer>
MultiOut(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_MULTIOUT, properties);
}

/**
 * @brief Helper function to create nnstreamer backbone layer
 */
inline std::unique_ptr<Layer>
BackboneNNStreamer(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_BACKBONE_NNSTREAMER, properties);
}

/**
 * @brief Helper function to create tflite backbone layer
 */
inline std::unique_ptr<Layer>
BackboneTFLite(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_BACKBONE_TFLITE, properties);
}

/**
 * @brief Helper function to create Embedding layer
 */
inline std::unique_ptr<Layer>
Embedding(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_EMBEDDING, properties);
}

/**
 * @brief Helper function to create RNN layer
 */
inline std::unique_ptr<Layer>
RNN(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_RNN, properties);
}

/**
 * @brief Helper function to create activation layer
 */
inline std::unique_ptr<Layer>
Activation(const std::string &act,
           const std::vector<std::string> &properties = {}) {
  std::vector<std::string> props(properties);
  props.push_back(act);
  return createLayer(LayerType::LAYER_ACTIVATION, props);
}

/**
 * @brief Helper function to create ReLU activation layer
 */
inline std::unique_ptr<Layer>
ReLU(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=relu", properties);
}

/**
 * @brief Helper function to create Tanh layer
 */
inline std::unique_ptr<Layer>
Tanh(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=tanh", properties);
}

/**
 * @brief Helper function to create sigmoid layer
 */
inline std::unique_ptr<Layer>
Sigmoid(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=sigmoid", properties);
}

/**
 * @brief Helper function to create softmax layer
 */
inline std::unique_ptr<Layer>
Softmax(const std::vector<std::string> &properties = {}) {
  return Activation("Activation=softmax", properties);
}

} // namespace layer

namespace loss {
/**
 * @brief Helper function to create mse layer
 */
std::unique_ptr<Layer> MSE(const std::vector<std::string> &properties = {});

/**
 * @brief Helper function to create cross entropy layer
 */
std::unique_ptr<Layer>
CrossEntropy(const std::vector<std::string> &properties = {});

} // namespace loss

/**
 * @brief General Layer Factory function to register Layer
 *
 * @param props property representation
 * @return std::unique_ptr<ml::train::Layer> created object
 */
template <typename T,
          std::enable_if_t<std::is_base_of<Layer, T>::value, T> * = nullptr>
std::unique_ptr<Layer> createLayer(const std::vector<std::string> &props = {}) {
  std::unique_ptr<Layer> ptr = std::make_unique<T>();

  if (ptr->setProperty(props) != ML_ERROR_NONE) {
    throw std::invalid_argument("Set properties failed for layer");
  }
  return ptr;
}

using CreateLayerFunc = Layer *(*)();
using DestroyLayerFunc = void (*)(Layer *);

/**
 * @brief  Layer Pluggable struct that enables pluggable layer
 *
 */
typedef struct {
  CreateLayerFunc createfunc;   /**< create layer function */
  DestroyLayerFunc destroyfunc; /**< destory function */
} LayerPluggable;

/**
 * @brief pluggable layer must have this structure defined
 */
extern "C" LayerPluggable ml_train_layer_pluggable;

} // namespace train
} // namespace ml
#endif // __ML_TRAIN_LAYER_H__
