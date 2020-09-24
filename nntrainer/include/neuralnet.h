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
#include <optimizer.h>
#include <pooling2d_layer.h>
#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Enumeration of Network Type
 *            0. KNN ( k Nearest Neighbor )
 *            1. REG ( Logistic Regression )
 *            2. NEU ( Neural Network )
 *            3. Unknown
 */
typedef enum { NET_KNN, NET_REG, NET_NEU, NET_UNKNOWN } NetType;

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
  friend class ModelLoader;

public:
  /**
   * @brief     Constructor of NeuralNetwork Class
   */
  NeuralNetwork(bool istrain = true) :
    is_train(istrain),
    batch_size(1),
    epochs(1),
    loss(0.0f),
    loss_type(LossType::LOSS_UNKNOWN),
    weight_initializer(WEIGHT_UNKNOWN),
    net_type(NET_UNKNOWN),
    data_buffer(NULL),
    continue_train(false),
    iter(0),
    initialized(false),
    def_name_count(0),
    loadedFromConfig(false) {}

  /**
   * @brief     Destructor of NeuralNetwork Class
   */
  ~NeuralNetwork() {}

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
    swap(lhs.is_train, rhs.is_train);
  }

  /**
   * @brief     Get Loss
   * @retval    loss value
   */
  float getLoss();

  /**
   * @brief     Set Optimizer
   * @retval    Optimizer
   */
  void setOptimizer(Optimizer optimizer) { opt = optimizer; };

  /**
   * @brief     Get Learning rate
   * @retval    Learning rate
   */
  float getLearningRate() { return opt.getLearningRate(); };

  /**
   * @brief     Set Loss
   * @param[in] l loss value
   */
  void setLoss(float l);

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
   * @brief     set Property/Configuration of Network for training after the
   * network has been initialized
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setTrainConfig(std::vector<std::string> values);

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
   * @brief     finalize NeuralNetwork Object
   */
  void finalize();

  /**
   * @brief     Run NeuralNetwork train
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train_run();

  /**
   * @brief     Run NeuralNetwork train
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train();

  /**
   * @brief     Run NeuralNetwork train
   * @param[in] values hyper parameters
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int train(std::vector<std::string> values);

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
   * @brief     check neural network is ready to init.
   * @retval #ML_ERROR_NONE neuralnetwork is ready to init
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to init.
   */
  int isInitializable();

  /**
   * @brief     check train or inference. Default is True (train)
   * @retval bool True : training Mode. False : Inference Mode
   */
  bool isTrain() { return is_train; };

  /**
   * @param[in] train true for train mode. false for inference mode
   * @brief     set Running Mode : Train or Inference
   */
  void setMode(bool train) { is_train = train; };

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
   * @brief print function for neuralnet
   * @param[in] out outstream
   * @param[in] flags verbosity from ml_train_summary_type_e
   */
  /// @todo: change print to use NeuralNetPrintOption and add way to print out
  void print(std::ostream &out, unsigned int flags = 0);

  /**
   * @brief print metrics function for neuralnet
   * @param[in] out outstream
   * @param[in] flags verbosity from ml_train_summary_type_e
   */
  void printMetrics(std::ostream &out, unsigned int flags = 0);

private:
  bool is_train; /**< is train or inference */

  unsigned int batch_size; /**< batch size */

  unsigned int epochs; /**< Maximum Epochs */

  float loss; /**< loss */

  LossType loss_type; /**< Loss Function type */

  WeightInitializer weight_initializer; /**< Weight Initialization type */

  std::string save_path; /**< Model path to save / read */

  Optimizer opt; /**< Optimizer, This gets copied into each layer, do not use
                    this directly */

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
   * @brief     Sets up and initialize the loss layer
   */
  int initLossLayer();

  /**
   * @brief     Realize act type to layer and insert it to layers
   * @param[in] ActiType act Activation Type
   * @param[in] int Position position to insert activation layer.
   * @note layer is inserted at position
   */
  int realizeActivationType(const ActiType act, const unsigned int position);

  /**
   * @copydoc int realizeActivationType(ActiType act, unsigned int &position);
   * @note layer is inserted at the back of layers
   */
  int realizeActivationType(const ActiType act);

  /**
   * @brief     Realize flatten type to layer and insert it to layers
   * @param[in] int Position position to insert the layer.
   * @note layer is inserted at position
   */
  int realizeFlattenType(const unsigned int position);

  /**
   * @copydoc int realizeActivationType(ActiType act, unsigned int &position);
   * @note layer is inserted at the back of layers
   */
  int realizeFlattenType();

  /**
   * @brief     Ensure that layer has a name
   */
  void ensureName(std::shared_ptr<Layer> layer, std::string prefix = "");
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __NEURALNET_H__ */
