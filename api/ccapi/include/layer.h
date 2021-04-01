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
  LAYER_LSTM,                                 /** LSTM Layer type */
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
   * @brief Get the layer type
   * @return const std::string type representation
   */
  virtual const std::string getType() const = 0;

  /**
   * @brief     Default allowed properties
   * - input shape : string
   * - bias zero : bool
   * - normalization : bool
   * - standardization : bool
   * - activation : string (type)
   * - epsilon : float
   * - weight_regularizer : string (type)
   * - weight_regularizer_constant : float
   * - unit : int
   * - weight_initializer : string (type)
   * - filter_size : int
   * - kernel_size : ( n , m )
   * - stride : ( n, m )
   * - padding : ( n, m )
   * - pool_size : ( n,m )
   * - pooling : max, average, global_max, global_average
   * - flatten : bool
   * - name : string (type)
   * - num_inputs : unsigned int (minimum 1)
   * - num_outputs : unsigned int (minimum 1)
   * - momentum : float,
   * - moving_mean_initializer : string (type),
   * - moving_variance_initializer : string (type),
   * - gamma_initializer : string (type),
   * - beta_initializer" : string (type)
   * - modelfile : model file for loading config for backbone layer
   * - input_layers : string (type)
   * - output_layers : string (type)
   * - trainable :
   * - flip_direction
   * - random_translate
   * - in_dim : int ( input dimension for embedding layer )
   * - out_dim : int ( output dimesion for embedding layer )
   * - in_length : int ( input length for embedding layer )
   * - recurrent_activation : string (type) - used only in lstm
   * - dist_layer : string (type) - layer name to be distributed
   */
  /**
   * @brief     set Property of layer
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual int setProperty(std::vector<std::string> values) = 0;

  /**
   * @brief     Get name of the layer
   * @retval    name of the layer
   * @note      This name is unique to this layer in a model
   * @Note      This name might be changed once this layer is added to the model
   * to keep the name unique to the model
   */
  virtual std::string getName() noexcept = 0;
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

/**
 * Aliases for various layers and losses
 */
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
 * @brief Helper function to create LSTM layer
 */
inline std::unique_ptr<Layer>
LSTM(const std::vector<std::string> &properties = {}) {
  return createLayer(LayerType::LAYER_LSTM, properties);
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

} // namespace train
} // namespace ml
#endif // __ML_TRAIN_LAYER_H__
