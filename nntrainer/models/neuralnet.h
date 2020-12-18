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

#include <map>
#include <memory>
#include <vector>
#ifdef PROFILE
#include <chrono>
#endif

#include <activation_layer.h>
#include <app_context.h>
#include <bn_layer.h>
#include <conv2d_layer.h>
#include <databuffer.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <layer_internal.h>
#include <loss_layer.h>
#include <manager.h>
#include <ml-api-common.h>
#include <network_graph.h>
#include <optimizer_internal.h>
#include <pooling2d_layer.h>
#include <tensor.h>

#include <model.h>
#include <nntrainer-api-common.h>

namespace nntrainer {

/**
 * @brief     Enumeration of Network Type
 */
using NetType = ml::train::ModelType;

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
class NeuralNetwork : public ml::train::Model {
  friend class ModelLoader; /** access private members of ModelLoader */

public:
  using NodeType = std::shared_ptr<Layer>; /** Type of a Node */
  using GraphType = std::vector<NodeType>; /** actual graph type */
  using FlatGraphType =
    std::vector<NodeType>; /** topological sorted, iterable 1-D list of nodes */
  using NetworkGraphType = nntrainer::NetworkGraph;

  /**
   * @brief     Constructor of NeuralNetwork Class
   */
  NeuralNetwork(AppContext app_context_ = AppContext(AppContext::Global()),
                bool bn_opt = true) :
    batch_size(1),
    epochs(1),
    epoch_idx(0),
    iter(0),
    loss(0.0f),
    loss_type(LossType::LOSS_UNKNOWN),
    weight_initializer(WeightInitializer::WEIGHT_UNKNOWN),
    net_type(NetType::UNKNOWN),
    data_buffer(nullptr),
    continue_train(false),
    initialized(false),
    compiled(false),
    def_name_count(0),
    loadedFromConfig(false),
    app_context(app_context_),
    in_place_bn_layer_optimization(bn_opt) {}

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

  int compile();

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
  int initialize();

  /**
   * @brief     Assign Graph Memory. This should be called after initialize.
   * @TODO      Consider to move Network Graph Class
   * @param[in] trainable Assign memory for inference or train mode
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int assignMem(bool trainable = true);

  /**
   * @brief     Update graph to make batch normalization in-place
   * @note      This assumes that the batch normalization implementation does
   * not need input/output of itself while backwarding. The reason is that the
   * batch normalization layer caches a processed form of its own input than the
   * input tensor itself.
   * @note      This optimization might break the working when some other
   * implementation of batch normalization layer is used or delegated to some
   * other backend. Ensure to verify this optimization with other
   * implementations once added.
   */
  void inPlaceBatchNormOptimization();

  /**
   * @brief     Forward Propagation of the neural network
   * @param[in] input List of Input Tensors taken by the neural network
   * @retval    List of Output Tensors
   */
  sharedConstTensors forwarding(sharedConstTensors input);

  /**
   * @brief     Forward Propagation of the neural network
   * @param[in] input List of Input Tensors taken by the neural network
   * @param[in] label List of Label Tensors for the model
   * @retval    List of Output Tensors
   */
  sharedConstTensors forwarding(sharedConstTensors input,
                                sharedConstTensors label);

  /**
   * @brief     Backward Propagation of the neural network
   * @param[in] input List of Input Tensors taken by the neural network
   * @param[in] label List of Label Tensors for the model
   * @param[in] iteration Iteration Number for the optimizer
   */
  void backwarding(sharedConstTensors label, int iteration);

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
  sharedConstTensors inference(sharedConstTensors X);

  /**
   * @brief     Run NeuralNetwork train with callback function by user
   * @param[in] dataset set the dataset
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setDataset(std::shared_ptr<ml::train::Dataset> dataset) {
    return setDataBuffer(std::static_pointer_cast<DataBuffer>(dataset));
  }

  /**
   * @brief     Run NeuralNetwork train with callback function by user
   * @param[in] databuffer set the databuffer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setDataBuffer(std::shared_ptr<DataBuffer> data_buffer);

  /**
   * @brief     add layer into neural network model
   * @param[in] layer layer to add
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLayer(std::shared_ptr<ml::train::Layer> layer) {
    return addLayer(std::static_pointer_cast<Layer>(layer));
  }

  /**
   * @brief     add layer into neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLayer(NodeType layer);

  /**
   * @brief     join passed graph into the existing graph model
   * @param[in] graph graph to be added/to extend
   * @param[in] prefix prefix added to names of layers from this graph
   * @note It is assumed that this model is valid by itself
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int extendGraph(GraphType graph, std::string prefix = "");

  /**
   * @brief     set optimizer for the neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setOptimizer(std::shared_ptr<ml::train::Optimizer> optimizer);

  /*
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int getLayer(const char *name, std::shared_ptr<ml::train::Layer> *layer);

  /*
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int getLayer(const char *name, NodeType *layer);

  /*
   * @brief     get input dimension of neural network
   * @retval std::vector<TensorDim> input dimension
   */
  std::vector<TensorDim> getInputDimension() {
    return model_graph.getInputDimension();
  }

  /*
   * @brief     get output dimension of neural network
   * @retval std::vector<TensorDim> output dimension
   */
  std::vector<TensorDim> getOutputDimension() {
    return model_graph.getOutputDimension();
  }

  /**
   * @brief get FlatGraph of current graph
   * @note flat graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval flatGraph of the current graph
   */
  FlatGraphType getFlatGraph() { return layers; }

  NetworkGraphType getNetworkGraph() { return model_graph; }

  /**
   * @brief get current graph from the model
   * @note graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval current graph
   */
  GraphType getGraph(const std::string &input_layer = "",
                     const std::string &output_layer = "");

  /**
   * @brief     Set loss type for the neural network.
   * @param[in] loss Type of the loss.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setLoss(LossType loss);

  /**
   * @brief Print Option when printing model info. The function delegates to the
   * `print`
   * @param out std::ostream to print
   * @param preset preset from `ml_train_summary_type_e`
   */
  void printPreset(std::ostream &out, unsigned int preset);

  /**
   * @brief Enable gradient memory sharing based optimization
   * @param opt True to enable, else false
   * @note This optimization has no performance overhead.
   */
  void setGradientMemoryOptimization(bool opt) {
    manager.setGradientMemoryOptimization(opt);
  }

  /**
   * @brief Enable in-place batch normalization layer operation
   * @param opt True to enable, else false
   * @note This optimization has no performance overhead.
   */
  void setInPlaceBNLayerOptimization(bool opt) {
    in_place_bn_layer_optimization = opt;
  }

/// @todo Make a more common class have this
/// Maybe appcontext can have this?
#ifdef PROFILE
  class Profiler {
  public:
    // Designated key.
    enum { FORWARD = 0 };

    /**
     * @brief query profile result
     *
     * @return std::chrono::time_point<std::chrono::steady_clock> profile result
     */
    std::chrono::milliseconds result(const int &key) { return time_taken[key]; }

    /**
     * @brief start profile
     *
     * @param key to record the profile result. Either designated key from enum
     * or arbitrary key can be used
     */
    void start(const int &key) {
      /// @todo: check if key is being reused with time_taken
      start_time[key] = std::chrono::steady_clock::now();
    }

    /**
     * @brief end profile
     *
     * @param key to record the profile result. Either designated key from enum
     * or arbitrary key can be used
     */
    void end(const int &key) {
      auto end = std::chrono::steady_clock::now();
      auto iter = start_time.find(key);

      if (iter == start_time.end()) {
        throw std::invalid_argument("profiler hasn't started with the key");
      }

      time_taken[key] = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - iter->second);
    }

  private:
    std::unordered_map<int, std::chrono::time_point<std::chrono::steady_clock>>
      start_time; /**< start_time of the clock */
    std::unordered_map<int, std::chrono::milliseconds> time_taken;
  };

public:
  /**
   * @brief Get the Profile Result
   *
   * @param key key to recorder the profile
   * @return std::chrono::time_point<std::chrono::steady_clock>
   */
  std::chrono::milliseconds getProfileResult(const int &key) {
    return profiler.result(key);
  }

private:
  Profiler profiler;
#endif

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

  unsigned int epoch_idx; /**< Number of epoch_idx  */

  unsigned int iter; /**< iterations trained */

  float loss; /**< loss */

  LossType loss_type; /**< Loss Function type */

  WeightInitializer weight_initializer; /**< Weight Initialization type */

  std::string save_path; /**< Model path to save / read */

  std::shared_ptr<Optimizer> opt; /**< Optimizer; this gets copied into each
                    layer, do not use this directly */

  NetType net_type; /**< Network Type */

  GraphType layers; /**< vector for store layer pointers */

  Manager manager; /**< nntrainer manager */

  std::shared_ptr<DataBuffer> data_buffer; /**< Data Buffer to get Input */

  bool continue_train; /**< Continue train from the previous state of
   optimizer and iterations */

  bool initialized; /**< Network is initialized */

  bool compiled; /**< Network is compiled */

  std::set<std::string>
    layer_names; /**< Set containing all the names of layers in the model */

  int def_name_count; /**< Count assigned to layer names declared by default */

  bool loadedFromConfig; /**< Check if config is loaded to prevent load twice */

  RunStats validation; /** validation statistics of the model */
  RunStats training;   /** training statistics of the model */
  RunStats testing;    /** testing statistics of the model */

  AppContext app_context; /** Configurations bound to current app */

  NetworkGraph model_graph; /** Network Model Graph */

  std::map<std::string, std::string>
    sub_in_out; /** This is map to identyfy input and output layer name of
                   subgraph */

  bool in_place_bn_layer_optimization; /**< Run batch normalization layer
                                          in-place */

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
   * @brief     check neural network is ready to compile.
   * @retval #ML_ERROR_NONE neuralnetwork is ready to compile
   * @retval #ML_ERROR_INVALID_PARAMETER not ready to compile.
   */
  int isCompilable();

  /**
   * @brief     Swap function for the class
   */
  friend void swap(NeuralNetwork &lhs, NeuralNetwork &rhs) {
    using std::swap;

    swap(lhs.batch_size, rhs.batch_size);
    swap(lhs.epochs, rhs.epochs);
    swap(lhs.epoch_idx, rhs.epoch_idx);
    swap(lhs.iter, rhs.iter);
    swap(lhs.loss, rhs.loss);
    swap(lhs.loss_type, rhs.loss_type);
    swap(lhs.weight_initializer, rhs.weight_initializer);
    swap(lhs.save_path, rhs.save_path);
    swap(lhs.opt, rhs.opt);
    swap(lhs.net_type, rhs.net_type);
    swap(lhs.layers, rhs.layers);
    swap(lhs.data_buffer, rhs.data_buffer);
    swap(lhs.continue_train, rhs.continue_train);
    swap(lhs.initialized, rhs.initialized);
    swap(lhs.model_graph, rhs.model_graph);
    swap(lhs.compiled, rhs.compiled);
    swap(lhs.sub_in_out, rhs.sub_in_out);
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

  /**
   * @brief Set the Save Path
   *
   * @param path path to set as a save path
   */
  void setSavePath(const std::string &path);
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __NEURALNET_H__ */
