/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	neuralnet.h
 * @date	04 December 2019
 * @brief	This is Neural Network Class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */
#ifndef __NEURALNET_H__
#define __NEURALNET_H__
#ifdef __cplusplus

#include <vector>

#include <activation_layer.h>
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <databuffer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <layer.h>
#include <loss_layer.h>
#include <ml-api-common.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <pooling2d_layer.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Enumeration of Network Type
 */
enum class NetType {
  NET_KNN,    /** k Nearest Neighbor */
  NET_REG,    /** Logistic Regression */
  NET_NEU,    /** Neural Network */
  NET_UNKNOWN /** Unknown */
};

/**
 * @brief     Statistics from running or training a model
 */
typedef struct RunStats_ {
  float accuracy; /** accuracy of the model */
  float loss;     /** loss of the model */

  RunStats_() : accuracy(0), loss(0) {}
} RunStats;

/**
 * @class   NeuralNetwork Class
 * @brief   NeuralNetwork Class which has Network Configuration & Layers
 */
class NeuralNetwork {
  friend class ModelLoader; /** access private members of ModelLoader */

public:
  /**
   * @brief     Constructor of NeuralNetwork Class
   */
  NeuralNetwork() :
    batch_size(1),
    epochs(1),
    loss(0.0f),
    loss_type(LossType::LOSS_UNKNOWN),
    weight_initializer(WeightInitializer::WEIGHT_UNKNOWN),
    net_type(NetType::NET_UNKNOWN),
    data_buffer(NULL),
    continue_train(false),
    iter(0),
    initialized(false),
    def_name_count(0),
    loadedFromConfig(false) {}

  /**
   * @brief     Destructor of NeuralNetwork Class
   */
  ~NeuralNetwork();

  /**
   * @brief     Get Loss from the previous ran batch of data
   * @retval    loss value
   */
  float getLoss();

  /**
   * @brief     Get Loss from the previous epoch of training data
   * @retval    loss value
   */
  float getTrainingLoss() { return training.loss; }

  /**
   * @brief     Get Loss from the previous epoch of validation data
   * @retval    loss value
   */
  float getValidationLoss() { return validation.loss; }

  /**
   * @brief     Get Learning rate
   * @retval    Learning rate
   */
  float getLearningRate() { return opt->getLearningRate(); };

  /**
   * @brief     Create and load the Network with ini configuration file.
   * @param[in] config config file path
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int loadFromConfig(std::string config);

  /**
   * @brief     set Property of Network
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setProperty(std::vector<std::string> values);

  /**
   * @brief     Initialize Network. This should be called after set all
   * hyperparameters.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int init();

  /**
   * @brief     Forward Propagation of the neural network
   * @param[in] input List of Input Tensors taken by the neural network
   * @retval    List of Output Tensors
   */
  sharedConstTensor forwarding(sharedConstTensor input);

  /**
   * @brief     Forward Propagation of the neural network
   * @param[in] input List of Input Tensors taken by the neural network
   * @param[in] label List of Label Tensors for the model
   * @retval    List of Output Tensors
   */
  sharedConstTensor forwarding(sharedConstTensor input,
                               sharedConstTensor label);

  /**
   * @brief     Backward Propagation of the neural network
   * @param[in] input List of Input Tensors taken by the neural network
   * @param[in] label List of Label Tensors for the model
   * @param[in] iteration Iteration Number for the optimizer
   */
  void backwarding(sharedConstTensor input, sharedConstTensor label,
                   int iteration);

  /**
   * @brief     save model and training parameters into file
   */
  void saveModel();

  /**
   * @brief     read model and training parameters from file
   */
  void readModel();

  /**
   * @brief     get Epochs
   * @retval    epochs
   */
  unsigned int getEpochs() { return epochs; };

  /**
   * @brief     Copy Neural Network
   * @param[in] from NeuralNetwork Object to copy
   * @retval    NeuralNewtork Object copyed
   */
  NeuralNetwork &copy(NeuralNetwork &from);

  /**
   * @brief     Run NeuralNetwork train
   * @param[in] values hyper parameters
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train(std::vector<std::string> values = {});

  /**
   * @brief     Run NeuralNetwork inference
   * @param[in] X input tensor
   * @retval shared_ptr<const Tensor>
   */
  sharedConstTensor inference(const Tensor X);

  /**
   * @brief     Run NeuralNetwork train with callback function by user
   * @param[in] train_func callback function to get train data. This provides
   * batch size data per every call.
   * @param[in] val_func callback function to get validation data. This provides
   * batch size data per every call.
   * @param[in] test_func callback function to get test data. This provides
   * batch size data per every call.
   * @param[in] values hyper-parameter list
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setDataBuffer(std::shared_ptr<DataBuffer> data_buffer);

  /**
   * @brief     add layer into neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLayer(std::shared_ptr<Layer> layer);

  /**
   * @brief     set optimizer for the neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setOptimizer(std::shared_ptr<Optimizer> optimizer);

  /*
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int getLayer(const char *name, std::shared_ptr<Layer> *layer);

  /*
   * @brief     get input dimension of neural network
   * @retval TensorDim input dimension
   */
  TensorDim getInputDimension() { return layers[0]->getInputDimension(); }

  /*
   * @brief     get output dimension of neural network
   * @retval TensorDim output dimension
   */
  TensorDim getOutputDimension() { return layers.back()->getOutputDimension(); }

  /**
   * @brief     Set loss type for the neural network.
   * @param[in] loss Type of the loss.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setLoss(LossType loss);

  enum class PropertyType {
    loss = 0,
    loss_type = 1,
    batch_size = 2,
    epochs = 3,
    save_path = 4,
    continue_train = 5,
    unknown = 6
  };

  /**
   * @brief Print Option when printing model info. The function delegates to the
   * `print`
   * @param out std::ostream to print
   * @param preset preset from `ml_train_summary_type_e`
   */
  void printPreset(std::ostream &out, unsigned int preset);

private:
  /**
   * @brief   Print Options when printing layer info
   */
  typedef enum {
    // clang-format off
  PRINT_INST_INFO  = (1 << 0), /**< Option to print type & instance address info */
  PRINT_GRAPH_INFO = (1 << 1), /**< Option to print graph topology info */
  PRINT_PROP       = (1 << 2), /**< Option to print properties */
  PRINT_OPTIMIZER  = (1 << 3), /**< Option to print optimizer */
  PRINT_METRIC       = (1 << 4), /**< Option to print if current network is set to training */
    // clang-format on
  } PrintOption;

  unsigned int batch_size; /**< batch size */

  unsigned int epochs; /**< Maximum Epochs */

  float loss; /**< loss */

  LossType loss_type; /**< Loss Function type */

  WeightInitializer weight_initializer; /**< Weight Initialization type */

  std::string save_path; /**< Model path to save / read */

  std::shared_ptr<Optimizer> opt; /**< Optimizer; this gets copied into each
                    layer, do not use this directly */

  NetType net_type; /**< Network Type */

  std::vector<std::shared_ptr<Layer>>
    layers; /**< vector for store layer pointers */

  std::shared_ptr<DataBuffer> data_buffer; /**< Data Buffer to get Input */

  bool continue_train; /**< Continue train from the previous state of optimizer
   and iterations */

  uint64_t iter; /**< Number of iterations trained */

  bool initialized; /**< Network is initialized */

  std::set<std::string>
    layer_names; /**< Set containing all the names of layers in the model */

  int def_name_count; /**< Count assigned to layer names declared by default */

  bool loadedFromConfig; /**< Check if config is loaded to prevent load twice */

  RunStats validation; /** validation statistics of the model */
  RunStats training;   /** training statistics of the model */
  RunStats testing;    /** testing statistics of the model */

  /**
   * @brief print function for neuralnet
   * @param[in] out outstream
   * @param[in] flags bit combination of Neuralnet::PrintOption
   * @param[in] Layer::PrintPreset print preset when to print layer properties
   */
  void print(
    std::ostream &out, unsigned int flags = 0,
    Layer::PrintPreset layerPrintPreset = Layer::PrintPreset::PRINT_SUMMARY);

  /**
   * @brief     Sets up and initialize the loss layer
   */
  int initLossLayer();

  /**
   * @brief     Set Loss
   * @param[in] l loss value
   */
  void setLoss(float l);

  /**
   * @brief     Run NeuralNetwork train
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train_run();

  /**
   * @brief     check neural network is ready to init.
   * @retval #ML_ERROR_NONE neuralnetwork is ready to init
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to init.
   */
  int isInitializable();

  /**
   * @brief     Realize act type to layer and insert it to layers
   * @param[in] ActivationType act Activation Type
   * @param[in] int Position position to insert activation layer.
   * @note layer is inserted at position
   */
  int realizeActivationType(const ActivationType act,
                            const unsigned int position);

  /**
   * @copydoc int realizeActivationType(ActivationType act, unsigned int
   * &position);
   * @note layer is inserted at the back of layers
   */
  int realizeActivationType(const ActivationType act);

  /**
   * @brief     Realize flatten type to layer and insert it to layers
   * @param[in] int Position position to insert the layer.
   * @note layer is inserted at position
   */
  int realizeFlattenType(const unsigned int position);

  /**
   * @copydoc int realizeActivationType(ActivationType act, unsigned int
   * &position);
   * @note layer is inserted at the back of layers
   */
  int realizeFlattenType();

  /**
   * @brief     Ensure that layer has a name
   */
  void ensureName(std::shared_ptr<Layer> layer, std::string prefix = "");

  /**
   * @brief     Swap function for the class
   */
  friend void swap(NeuralNetwork &lhs, NeuralNetwork &rhs) {
    using std::swap;

    swap(lhs.batch_size, rhs.batch_size);
    swap(lhs.epochs, rhs.epochs);
    swap(lhs.loss, rhs.loss);
    swap(lhs.loss_type, rhs.loss_type);
    swap(lhs.weight_initializer, rhs.weight_initializer);
    swap(lhs.save_path, rhs.save_path);
    swap(lhs.opt, rhs.opt);
    swap(lhs.net_type, rhs.net_type);
    swap(lhs.layers, rhs.layers);
    swap(lhs.data_buffer, rhs.data_buffer);
    swap(lhs.continue_train, rhs.continue_train);
    swap(lhs.iter, rhs.iter);
    swap(lhs.initialized, rhs.initialized);
    swap(lhs.layer_names, rhs.layer_names);
    swap(lhs.def_name_count, rhs.def_name_count);
    swap(lhs.loadedFromConfig, rhs.loadedFromConfig);
  }

  /**
   * @brief     set Property/Configuration of Network for training after the
   * network has been initialized
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setTrainConfig(std::vector<std::string> values);

  /**
   * @brief     Update batch size of the model as well as its layers/dataset
   */
  void setBatchSize(unsigned int batch_size);

  /**
   * @brief print metrics function for neuralnet
   * @param[in] out outstream
   * @param[in] flags verbosity from ml_train_summary_type_e
   */
  void printMetrics(std::ostream &out, unsigned int flags = 0);
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __NEURALNET_H__ */
