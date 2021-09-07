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

#include <array>
#include <map>
#include <memory>
#include <tuple>
#include <vector>
#ifdef PROFILE
#include <chrono>
#endif

#include <app_context.h>
#include <dynamic_training_optimization.h>
#include <layer_node.h>
#include <ml-api-common.h>
#include <model_common_properties.h>
#include <network_graph.h>
#include <optimizer_devel.h>
#include <tensor.h>

#include <model.h>
#include <nntrainer-api-common.h>

namespace ml::train {
class DataSet;
enum class DatasetType;
enum class DatasetModeType;
} // namespace ml::train

namespace nntrainer {

/**
 * @brief     Enumeration of Network Type
 */
using NetType = ml::train::ModelType;

class DataBuffer;
using DatasetType = ml::train::DatasetType;
using DatasetModeType = ml::train::DatasetModeType;
/**
 * @brief     Statistics from running or training a model
 */
struct RunStats {
  float accuracy;     /** accuracy of the model */
  float loss;         /** loss of the model */
  int num_iterations; /** number of iterations done on this stat */
  unsigned int
    num_correct_predictions; /** number of right sample on this run */

  RunStats() :
    accuracy(0),
    loss(0),
    num_iterations(0),
    num_correct_predictions(0) {}
};

/**
 * @class   NeuralNetwork Class
 * @brief   NeuralNetwork Class which has Network Configuration & Layers
 */
class NeuralNetwork : public ml::train::Model {
  friend class ModelLoader; /** access private members of ModelLoader */

public:
  using NodeType = std::shared_ptr<LayerNode>; /** Type of a Node */
  using GraphType = std::vector<NodeType>;     /** actual graph type */
  using FlatGraphType =
    std::vector<NodeType>; /** topological sorted, iterable 1-D list of nodes */
  using NetworkGraphType = nntrainer::NetworkGraph;

  /**
   * @brief     Constructor of NeuralNetwork Class
   */
  NeuralNetwork(AppContext app_context_ = AppContext(AppContext::Global()),
                bool in_place_opt = true);

  /**
   * @brief     Destructor of NeuralNetwork Class
   */
  ~NeuralNetwork();

  /**
   * @brief     Set labels to the layers which require the label
   * @todo      Set label with multiple labels
   * @param     label label
   */
  void setLabels(sharedConstTensors label);

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
  int loadFromConfig(const std::string &config);

  /**
   * @brief     Compile the graph in the model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int compile();

  /**
   * @brief     set Property of Network
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief     Initialize Network. This should be called after set all
   * hyperparameters.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize();

  /**
   * @brief     Allocate memory for the model. This should be called after
   * initialize.
   * @param[in] trainable Assign memory for inference or train mode
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int allocate(bool trainable = true);

  /**
   * @brief     Deallocate memory for the model.
   * @param[in] trainable Assign memory for inference or train mode
   * @retval #ML_ERROR_NONE Successful.
   * @note This does not free the model graph but only the weight tensors, and
   * input/output/gradient/derivative tensors if any.
   */
  int deallocate();

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
  void inPlaceOptimization(const std::string &layer_type);

  /**
   * @brief     Forward Propagation of the neural network
   */
  sharedConstTensors forwarding(bool training = true);

  /**
   * @brief     Forward Propagation of the neural network
   * @param[in] input List of Input Tensors taken by the neural network
   * @param[in] label List of Label Tensors for the model
   * @retval    List of Output Tensors
   */
  sharedConstTensors forwarding(sharedConstTensors input,
                                sharedConstTensors label = {},
                                bool training = true);

  /**
   * @brief     Backward Propagation of the neural network
   * @param[in] label List of Label Tensors for the model
   * @param[in] iteration Iteration Number for the optimizer
   */
  void backwarding(sharedConstTensors label, int iteration);

  /**
   * @brief     Backward Propagation of the neural network
   * @param[in] iteration Iteration Number for the optimizer
   */
  void backwarding(int iteration);

  /**
   * @copydoc Model::save(const std::string &file_path, ml::train::ModelFormat
   * format);
   */
  void save(const std::string &file_path,
            ml::train::ModelFormat format =
              ml::train::ModelFormat::MODEL_FORMAT_BIN) override;

  /**
   * @copydoc Model::load(const std::string &file_path, ml::train::ModelFormat
   * format);
   */
  void load(const std::string &file_path,
            ml::train::ModelFormat format =
              ml::train::ModelFormat::MODEL_FORMAT_BIN) override;

  /**
   * @brief     get Epochs
   * @retval    epochs
   */
  unsigned int getEpochs() {
    return std::get<props::Epochs>(model_flex_props);
  };

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
  int train(const std::vector<std::string> &values = {}) override;

  /**
   * @brief     Run NeuralNetwork inference
   * @param[in] X input tensor
   * @retval shared_ptr<const Tensor>
   */
  sharedConstTensors inference(sharedConstTensors X, bool free_mem = true);

  /**
   * @brief     Run the inference of the model
   * @param[in] input inputs as a list of each input data
   * @param[in] batch batch size of current input
   * @retval list of output as float *
   * @note The output memory must not be freed by the caller
   */
  std::vector<float *> inference(std::vector<float *> &input,
                                 unsigned int batch);

  /**
   * @brief     Run NeuralNetwork train with callback function by user
   * @param[in] dt datatype (mode) where it should be
   * @param[in] dataset set the dataset
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setDataset(const DatasetModeType &dt,
                 std::shared_ptr<ml::train::Dataset> dataset);

  /**
   * @brief     Run NeuralNetwork train with callback function by user
   * @param[in] dt datatype (mode) where it should be
   * @param[in] databuffer set the databuffer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int setDataBuffer(const DatasetModeType &dt,
                    std::shared_ptr<DataBuffer> data_buffer);

  /**
   * @brief     add layer into neural network model
   * @param[in] layer layer to add
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int addLayer(std::shared_ptr<ml::train::Layer> layer) {
    return addLayer(std::static_pointer_cast<LayerNode>(layer));
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

  /**
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int getLayer(const char *name,
                       std::shared_ptr<ml::train::Layer> *layer);

  /**
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int getLayer(const char *name, NodeType *layer);

  /**
   * @brief     get input dimension of neural network
   * @retval std::vector<TensorDim> input dimension
   */
  std::vector<TensorDim> getInputDimension() {
    if (!compiled) {
      throw std::logic_error("model should be compiled before get dimension");
    }
    return model_graph.getInputDimension();
  }

  /**
   * @brief     get output dimension of neural network
   * @retval std::vector<TensorDim> output dimension
   */
  std::vector<TensorDim> getOutputDimension() {
    if (!compiled) {
      throw std::logic_error("model should be compiled before get dimension");
    }
    return model_graph.getOutputDimension();
  }

  /**
   * @brief get FlatGraph of current graph
   * @note flat graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval flatGraph of the current graph
   * @note these layers will be in sorted order if the model is compiled,
   * otherwise the order is the order of addition of layers in the model.
   */
  FlatGraphType getFlatGraph() { return model_graph.getLayerNodes(); }

  /**
   * @brief get if the model is empty
   * @param[out] true if empty, else false
   */
  bool empty() const { return model_graph.empty(); }

  /**
   * @brief get the number of nodes in the model
   * @param[out] number of nodes
   */
  size_t size() const { return model_graph.size(); }

  /**
   * @brief     get network graph
   * @retval NetowrkGraphType
   */
  NetworkGraphType getNetworkGraph() { return model_graph; }

  /**
   * @brief get current graph from the model
   * @note graph contains pointer to the actual nodes, which is not deeply
   * copied.
   * @retval current graph
   */
  GraphType getUnsortedLayers(const std::string &input_layer = "",
                              const std::string &output_layer = "");

  /**
   * @brief     Summarize the model
   * @param out std::ostream to get the model summary
   * @param verbosity verbosity of the summary
   */
  virtual void summarize(std::ostream &out, ml_train_summary_type_e verbosity) {
    printPreset(out, (unsigned int)verbosity);
  }

  /**
   * @brief Print Option when printing model info. The function delegates to the
   * `print`
   * @param out std::ostream to print
   * @param preset preset from `ml_train_summary_type_e`
   */
  virtual void printPreset(std::ostream &out, unsigned int preset);

  /**
   * @brief Enable gradient memory sharing based optimization
   * @param opt True to enable, else false
   * @note This optimization has no performance overhead.
   */
  void setGradientMemoryOptimization(bool opt) {
    model_graph.setGradientMemoryOptimization(opt);
  }

  /**
   * @brief Enable derivative memory sharing based optimization
   * @param opt True to enable, else false
   * @note This optimization has no performance overhead.
   */
  void setDerivativeMemoryOptimization(bool opt) {
    model_graph.setDerivativeMemoryOptimization(opt);
    if (opt == false)
      setInPlaceLayerOptimization(false);
  }

  /**
   * @brief Enable in-place layer operations
   * @param opt True to enable, else false
   * @note This optimization has no performance overhead.
   */
  void setInPlaceLayerOptimization(bool opt) {
    in_place_optimization = opt;
    model_graph.setInPlaceActivationOptimization(opt);
  }

  /**
   * @brief Enable inout memory sharing based optimization for inference
   * @param opt True to enable, else false
   * @note This optimization has no performance overhead.
   */
  void setInferenceInOutMemoryOptimization(bool opt) {
    model_graph.setInferenceInOutMemoryOptimization(opt);
  }

  /**
   * @brief Enable dynamic fine-tuning optimization
   * @param threshold Comparison limit to decide if weight updated or not
   * @param mode dynamic fine-tuning optimization mode. Supported modes are
   * "max" and "norm" for now
   */
  void enableDynamicTraining(
    float threshold, std::string op = DynamicTrainingOptimization::dft_opt_norm,
    std::string mode = DynamicTrainingOptimization::dft_opt_mode_derivative) {
    dynamic_training_opt.setThreshold(threshold);
    dynamic_training_opt.setOp(op);
    dynamic_training_opt.setMode(mode);
    dynamic_training_opt.enable();
  }

  /**
   * @brief Disable dynamic fine-tuning optimization
   */
  void disableDynamicFineTuning() { dynamic_training_opt.disable(); }

private:
  using FlexiblePropTypes =
    std::tuple<props::Epochs, props::TrainingBatchSize, props::SavePath, props::ContinueTrain>;
  using RigidPropTypes = std::tuple<props::LossType>;

  RigidPropTypes model_props;         /**< model props */
  FlexiblePropTypes model_flex_props; /**< model train props */
  std::string load_path; /**< path to load weights when initialize  */

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

  unsigned int epoch_idx; /**< Number of epoch_idx  */

  unsigned int iter; /**< iterations trained */

  float loss; /**< loss */

  std::shared_ptr<Optimizer> opt; /**< Optimizer; this gets copied into each
                    layer, do not use this directly */

  std::array<std::shared_ptr<DataBuffer>, 3>
    data_buffers; /**< Data Buffers to get Input */

  bool initialized; /**< Network is initialized */

  bool compiled; /**< Network is compiled */

  bool loadedFromConfig; /**< Check if config is loaded to prevent load twice */

  RunStats validation; /** validation statistics of the model */
  RunStats training;   /** training statistics of the model */
  RunStats testing;    /** testing statistics of the model */

  AppContext app_context; /** Configurations bound to current app */

  NetworkGraph model_graph; /** Network Model Graph */

  bool in_place_optimization; /**< Run batch normalization, activation, etc
                                 layers in-place */

  DynamicTrainingOptimization dynamic_training_opt; /**< Dynamic fine-tuning
   optimization mode. supported modes are "max" and "norm" */

  /**
   * @brief save model in ini
   *
   * @param file_path file path
   */
  void saveModelIni(const std::string &file_path);

  /**
   * @brief print function for neuralnet
   * @param[in] out outstream
   * @param[in] flags bit combination of Neuralnet::PrintOption
   * @param[in] Layer::PrintPreset print preset when to print layer properties
   */
  void print(std::ostream &out, unsigned int flags = 0,
             LayerNode::PrintPreset layerPrintPreset =
               LayerNode::PrintPreset::PRINT_SUMMARY);

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
   * @brief     Swap function for the class
   */
  friend void swap(NeuralNetwork &lhs, NeuralNetwork &rhs);

  /**
   * @brief     set Property/Configuration of Network for training after the
   * network has been initialized
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  void setTrainConfig(const std::vector<std::string> &values);

  /**
   * @brief print metrics function for neuralnet
   * @param[in] out outstream
   * @param[in] flags verbosity from ml_train_summary_type_e
   */
  void printMetrics(std::ostream &out, unsigned int flags = 0);

  /**
   * @brief     Match the given tensor shape with input shape of the model
   * @param[in] X input tensor
   * @retval true if matches, false is error
   */
  bool validateInput(sharedConstTensors X);

  /**
   * @brief     Backward Propagation for the layer
   * @param[in] layer Layer to backpropagate
   * @param[in] iteration Iteration Number for the optimizer
   * @param[in] calc_derivative If the derivative for previous layer must be
   * calculated
   */
  void backwarding(std::shared_ptr<LayerNode> node, int iteration,
                   bool calc_derivative);
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __NEURALNET_H__ */
