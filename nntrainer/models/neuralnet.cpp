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
 * @file	neuralnet.cpp
 * @date	04 December 2019
 * @brief	This is Neural Network Class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include <databuffer_file.h>
#include <databuffer_func.h>
#include <model_loader.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <profiler.h>
#include <time_dist.h>
#include <unordered_set>
#include <util_func.h>

/**
 * @brief Internal enum values for nntrainer to summarize model accuracy & loss
 */
#define ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS 101
#define ML_TRAIN_SUMMARY_MODEL_VALID_LOSS 102
#define ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY 103

namespace nntrainer {

int NeuralNetwork::loadFromConfig(std::string config) {
  if (loadedFromConfig == true) {
    ml_loge("cannnot do loadFromConfig twice");
    return ML_ERROR_INVALID_PARAMETER;
  }

  ModelLoader loader(app_context);
  NeuralNetwork tempNet(*this);
  int status = loader.loadFromConfig(config, tempNet);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  tempNet.loadedFromConfig = true;
  swap(tempNet, *this);

  return ML_ERROR_NONE;
}

int NeuralNetwork::setProperty(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseNetProperty(key);

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::loss: {
      status = setFloat(loss, value);
      NN_RETURN_STATUS();
    } break;
    case PropertyType::loss_type: {
      status = setLoss(value);
      NN_RETURN_STATUS();
    } break;
    default:
      status = setTrainConfig({values[i]});
      NN_RETURN_STATUS();
      break;
    }
  }

  return status;
}

int NeuralNetwork::setTrainConfig(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    status = getKeyValue(values[i], key, value);
    NN_RETURN_STATUS();

    unsigned int type = parseNetProperty(key);

    switch (static_cast<PropertyType>(type)) {
    case PropertyType::epochs: {
      status = setUint(epochs, value);
      NN_RETURN_STATUS();
    } break;
    case PropertyType::save_path: {
      save_path = value;
    } break;
    case PropertyType::continue_train: {
      bool cont_train;
      status = setBoolean(cont_train, value);
      NN_RETURN_STATUS();
      continue_train = cont_train;
      opt->setProperty({values[i]});
    } break;
    case PropertyType::batch_size: {
      status = setUint(batch_size, value);
      NN_RETURN_STATUS();
      if (initialized)
        setBatchSize();
      /** TODO: increase buffer size if it is smaller than batch size.
       * also if this is set with default batch size, then make it
       * smaller/larger
       */
    } break;
    default:
      ml_loge("Error: Unknown Network Property Key");
      status = ML_ERROR_INVALID_PARAMETER;
      return status;
    }
  }

  return status;
}

int NeuralNetwork::compile() {
  int status = model_graph.compile(loss_type);
  NN_RETURN_STATUS();

  compiled = true;

  return status;
}

int NeuralNetwork::initialize() {
  int status = ML_ERROR_NONE;

  if (initialized) {
    ml_loge("Error: Initializing the model again");
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (!compiled) {
    ml_loge("Error: Need to compile first");
    return ML_ERROR_NOT_SUPPORTED;
  }

  unsigned int n_layers = (unsigned int)model_graph.size();

  ml_logd("initializing neural network, layer size: %d", n_layers);

  setBatchSize();

  status = model_graph.initialize(manager);
  NN_RETURN_STATUS();

  // initialize optimizer and related variables
  if (opt) {
    opt->initialize();
    std::function<std::vector<TensorDim>(const TensorDim &)> cb =
      [this](const TensorDim &dim) {
        return opt->getOptimizerVariableDim(dim);
      };
    manager->requestOptimizerVariable(cb, true);
  }

  // Allocate and initialize weights
  manager->initializeWeights();
  manager->allocateWeights();

  if (in_place_optimization) {
    model_graph.inPlaceOptimize(*manager);
  }

  initialized = true;
  return status;
}

/**
 * @brief     free layers
 */
NeuralNetwork::~NeuralNetwork() {
  manager.reset();
  model_graph.reset();

  if (data_buffer) {
    data_buffer->clear();
  }
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(bool training) {
  return model_graph.forwarding(training);
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(sharedConstTensors input,
                                             sharedConstTensors label,
                                             bool training) {

  NNTR_THROW_IF(input[0]->batch() != batch_size ||
                  (!label.empty() && label[0]->batch() != batch_size),
                std::logic_error)
    << "Error: mismatch in batchsize for data and model."
    << " input_batch: " << input[0]->batch()
    << " label_batch: " << label[0]->batch() << " target_batch: " << batch_size;

  auto fill_label = [&label](auto const &layer_node) {
    NNTR_THROW_IF(label.size() != layer_node->getOutputDimensions().size(),
                  std::invalid_argument)
      << "label size does not match with the layer requirements"
      << " layer: " << layer_node->getName() << " label size: " << label.size()
      << " requirements size: " << layer_node->getNumOutputs();

    for (unsigned int i = 0; i < layer_node->getOutputDimensions().size(); i++) {
      layer_node->getOutputGrad(i) = *label[i];
    }
  };

  auto clear_label = [](auto const &layer_node) {
    for (unsigned int i = 0; i < layer_node->getOutputDimensions().size(); i++) {
      layer_node->getOutputGrad(i) = Tensor();
    }
  };

  /// feed or clear label
  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    auto const &lnode = *iter;
    if (lnode->requireLabel()) {
      label.empty() ? clear_label(lnode) : fill_label(lnode);
    }
  }

  model_graph.getSortedLayerNode(0)->getInput(0) = *input[0].get();

  return forwarding(training);
}

void NeuralNetwork::backwarding(std::shared_ptr<LayerNode> node, int iteration,
                                bool calc_derivative) {
  /**
   * Do not change this order:
   * 1. calcGradient
   * 2. calcDerivative
   * 3. applyGradient
   */
  bool apply_gradient = true;
  /** If gradient optimization mode, then calculate gradient first */
  if (dynamic_training_opt.isGradientMode())
    node->calcGradient();

  /**
   * If optimization off, or gradient must be applied, then this will be true
   */
  // auto &layer = node->getObject();
  // apply_gradient = dynamic_training_opt.checkIfApply(
  //   layer->getWeightsRef(), layer->net_input[0], layer->net_hidden[0], opt,
  //   iteration);

  /** If gradient must be applied and its not gradient mode, calculate gradient
   */
  if (!dynamic_training_opt.isGradientMode() && apply_gradient)
    node->calcGradient();

  if (calc_derivative)
    node->calcDerivative();

  if (apply_gradient && node->getTrainable()) {
    // TODO: ask network_graph for weights of node and then remove
    // getWeightObject() interface from layer_context
    for (unsigned int idx = 0; idx < node->getNumWeights(); idx++) {
      auto &weight = node->getWeightObject(idx);
      if (weight.hasGradient()) {
        weight.calcRegularizationGradient();
        opt->applyGradient(weight, iteration);
      }
    }
  }
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(int iteration) {
  /**
   * last layer backwarding is run out of this loop
   */
  auto iter_begin = model_graph.getBackwardingBeginIter();
  auto iter_end = model_graph.getBackwardingEndIter();

  /// there is no layer to train, so backwarding is essentially noop
  if (iter_begin == iter_end) {
    return;
  }

  auto const &lptr_begin = (*iter_begin);

  if (lptr_begin->requireLabel() == false)
    throw std::runtime_error(
      "Error: last layer does not accept label, we can't train");

  auto iter = iter_begin;
  for (; iter != iter_end - 1; iter++) {
    backwarding(*iter, iteration, true);
  }

  /**
   * The last trainable layer need not calculate the derivatives
   */
#ifdef ENABLE_TEST
  backwarding(*iter, iteration, true);
#else
  backwarding(*iter, iteration, false);
#endif
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(sharedConstTensors label, int iteration) {
  auto const &loss_layer_node =
    model_graph.getSortedLayerNode(model_graph.size() - 1);
  loss_layer_node->getOutputGrad(0) = *label[0].get();

  backwarding(iteration);
}

float NeuralNetwork::getLoss() {
  loss = 0.0f;

  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    loss += (*iter)->getLoss();
  }
  return loss;
}

void NeuralNetwork::setLoss(float l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    batch_size = from.batch_size;
    loss = from.loss;
    opt = from.opt;

    model_graph.copy(from.model_graph);
  }
  return *this;
}

/**
 * @brief     save model to file
 *            save Weight & Bias Data into file by calling save from layer
 *            save training parameters from the optimizer
 */
void NeuralNetwork::saveModel() {
  if (!initialized)
    throw std::runtime_error("Cannot save the model before initialize.");

  if (save_path == std::string()) {
    return;
  }

  if (!initialized)
    throw std::runtime_error("Cannot save the model before initialize.");

  std::ofstream model_file(save_path, std::ios::out | std::ios::binary);

  NNTR_THROW_IF(!model_file.good(), std::invalid_argument)
    << "model file not opened, file path: " << save_path
    << " reason: " << strerror(errno);

  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    (*iter)->save(model_file);
  }
  model_file.write((char *)&epoch_idx, sizeof(epoch_idx));
  model_file.write((char *)&iter, sizeof(iter));
  model_file.close();
}

/**
 * @brief     read model from file
 *            read Weight & Bias Data into file by calling save from layer
 *            read training parameters from the optimizer if continuing train
 */
void NeuralNetwork::readModel() {
  if (!initialized)
    throw std::runtime_error("Cannot read the model before initialize.");

  if (save_path == std::string()) {
    return;
  }

  if (!isFileExist(save_path)) {
    ml_logd("skipping reading model, path is not valid: %s", save_path.c_str());
    return;
  }

  if (!initialized)
    throw std::runtime_error("Cannot save the model before initialize.");

  NeuralNetwork tmp(*this);

  std::ifstream model_file(save_path, std::ios::in | std::ios::binary);

  for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
    (*iter)->read(model_file);
  }

  checkedRead(model_file, (char *)&tmp.epoch_idx, sizeof(epoch_idx),
              "[NeuralNetwork::readModel] failed to read epoch_idx");
  checkedRead(model_file, (char *)&tmp.iter, sizeof(iter),
              "[NeuralNetwork::readModel] failed to read iteration");

  model_file.close();
  ml_logi("read modelfile: %s", save_path.c_str());

  swap(tmp, *this);
}

void NeuralNetwork::setBatchSize(unsigned int batch) {
  batch_size = batch;

  model_graph.setBatchSize(batch);
  manager->setBatchSize(batch);

  if (data_buffer && data_buffer->setBatchSize(batch_size) != ML_ERROR_NONE)
    throw std::invalid_argument("Error setting batchsize for the dataset");
}

bool NeuralNetwork::validateInput(sharedConstTensors X) {

  auto const &first_layer_node = model_graph.getSortedLayerNode(0);
  auto input_dim = first_layer_node->getInputDimensions();
  if (X.size() != input_dim.size()) {
    ml_loge("Error: provided number of inputs %d, required %d", (int)X.size(),
            (int)input_dim.size());
    return false;
  }

  for (unsigned int dim = 0; dim < input_dim.size(); dim++) {
    if (input_dim[dim] != X[dim]->getDim()) {
      ml_loge("Error: provided input shape does not match required shape");
      std::stringstream ss;
      ss << X[dim]->getDim();
      ml_loge("Provided tensor summary : %s", ss.str().c_str());

      ss.str(std::string());
      ss << input_dim[dim];
      ml_loge("Required tensor summary : %s", ss.str().c_str());
      return false;
    }
  }

  return true;
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X,
                                            bool free_mem) {
  if (batch_size != X[0]->batch()) {
    /**
     * Note that inference resets batch_size of the previous train configuration
     * Next train must set its batch_size if inference is run with this model.
     */
    setBatchSize(X[0]->batch());
  }

  sharedConstTensors out;
  if (!validateInput(X))
    throw std::invalid_argument("Input validation failed.");

  allocate(false);

  START_PROFILE(profile::NN_FORWARD);
  forwarding(X, {}, false);
  END_PROFILE(profile::NN_FORWARD);

  auto const &last_layer_node =
    model_graph.getSortedLayerNode(model_graph.size() - 1);
  for (unsigned int i = 0; i < last_layer_node->getNumOutputs(); ++i) {
    out.push_back(MAKE_SHARED_TENSOR(last_layer_node->getOutput(i)));
  }

  if (free_mem)
    /**
     * Free the memory needed for training before exiting.
     * Note that this does not free the weights for the model.
     * Weights of the model will be freed when the model is destroyed.
     */
    manager->deallocateTensors(false);

  return out;
}

int NeuralNetwork::allocate(bool trainable) {
  // TODO: directly replace this
  manager->initializeTensors(trainable);

  manager->allocateTensors();
  return ML_ERROR_NONE;
}

int NeuralNetwork::deallocate() {
  manager->deallocateTensors(true);

  return ML_ERROR_NONE;
}

int NeuralNetwork::train(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  if (data_buffer == nullptr) {
    ml_loge("Cannot initialize the model without the data buffer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!opt) {
    ml_loge("Cannot train network without optimizer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = setTrainConfig(values);
  NN_RETURN_STATUS();

  /** set batch size just before training */
  setBatchSize(batch_size);

  status = allocate(true);
  NN_RETURN_STATUS();

  /** Setup data buffer properties */
  status = data_buffer->setClassNum(getOutputDimension()[0].width());
  NN_RETURN_STATUS();

  status = data_buffer->setFeatureSize(getInputDimension()[0]);
  NN_RETURN_STATUS();

  status = data_buffer->init();
  NN_RETURN_STATUS();

  status = train_run();

  /**
   * Free the memory needed for training before exiting.
   * Note that this does not free the weights for the model.
   * Weights of the model will be freed when the model is destroyed.
   */
  manager->deallocateTensors(false);
  return status;
}

/**
 * @brief     Run NeuralNetwork train with callback function by user
 */
int NeuralNetwork::train_run() {
  int status = ML_ERROR_NONE;

  if (!continue_train) {
    epoch_idx = 0;
    iter = 0;
  }

  auto const &first_layer_node = model_graph.getSortedLayerNode(0);
  auto const &last_layer_node =
    model_graph.getSortedLayerNode(model_graph.size() - 1);

  auto &output = last_layer_node->getOutput(0);
  auto &label = last_layer_node->getOutputGrad(0);
  auto &in = first_layer_node->getInput(0);

  for (epoch_idx = epoch_idx + 1; epoch_idx <= epochs; ++epoch_idx) {
    training.loss = 0.0f;
    status = data_buffer->run(nntrainer::BufferType::BUF_TRAIN);
    if (status != ML_ERROR_NONE) {
      data_buffer->clear(BufferType::BUF_TRAIN);
      return status;
    }

    if (data_buffer->getValidation()[(int)nntrainer::BufferType::BUF_TEST]) {
      status = data_buffer->run(nntrainer::BufferType::BUF_TEST);
      if (status != ML_ERROR_NONE) {
        data_buffer->clear(BufferType::BUF_TEST);
        return status;
      }
    }

    int count = 0;

    while (true) {
      if (data_buffer->getDataFromBuffer(nntrainer::BufferType::BUF_TRAIN,
                                         in.getData(), label.getData())) {
        try {
          forwarding(true);
          backwarding(iter++);
        } catch (std::exception &e) {
          data_buffer->clear(nntrainer::BufferType::BUF_TRAIN);
          ml_loge("Error: training error in #%d/%d. %s", epoch_idx, epochs,
                  e.what());
          throw;
        }
        std::cout << "#" << epoch_idx << "/" << epochs;
        float loss = getLoss();
        data_buffer->displayProgress(count++, nntrainer::BufferType::BUF_TRAIN,
                                     loss);
        training.loss += loss;
      } else {
        data_buffer->clear(nntrainer::BufferType::BUF_TRAIN);
        break;
      }
    }

    if (count == 0)
      throw std::runtime_error("No training data");

    training.loss /= count;
    saveModel();

    std::cout << "#" << epoch_idx << "/" << epochs
              << " - Training Loss: " << training.loss;

    if (data_buffer->getValidation()[(int)nntrainer::BufferType::BUF_VAL]) {
      int right = 0;
      validation.loss = 0.0f;
      unsigned int tcases = 0;

      status = data_buffer->run(nntrainer::BufferType::BUF_VAL);
      if (status != ML_ERROR_NONE) {
        data_buffer->clear(BufferType::BUF_VAL);
        return status;
      }

      while (true) {
        if (data_buffer->getDataFromBuffer(nntrainer::BufferType::BUF_VAL,
                                           in.getData(), label.getData())) {
          forwarding(false);
          auto model_out = output.argmax();
          auto label_out = label.argmax();
          for (unsigned int b = 0; b < batch_size; b++) {
            if (model_out[b] == label_out[b])
              right++;
          }
          validation.loss += getLoss();
          tcases++;
        } else {
          data_buffer->clear(nntrainer::BufferType::BUF_VAL);
          break;
        }
      }

      if (tcases == 0) {
        ml_loge("Error : 0 test cases");
        status = ML_ERROR_INVALID_PARAMETER;
        return status;
      }
      validation.loss /= (float)(tcases);
      validation.accuracy = right / (float)(tcases * batch_size) * 100.0f;
      std::cout << " >> [ Accuracy: " << validation.accuracy
                << "% - Validation Loss : " << validation.loss << " ] ";
    }
    std::cout << std::endl;
  }

  return status;
}

int NeuralNetwork::addLayer(NodeType layer) {
  int status = ML_ERROR_NONE;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  /** Insert the layer to the graph */
  model_graph.addLayer(layer);

  return status;
}

int NeuralNetwork::extendGraph(GraphType graph, std::string prefix) {
  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (graph.size() == 0)
    return ML_ERROR_NONE;

  model_graph.extendGraph(graph, prefix);
  return ML_ERROR_NONE;
}

NeuralNetwork::GraphType
NeuralNetwork::getUnsortedLayers(const std::string &input_layer,
                                 const std::string &output_layer) {
  return model_graph.getUnsortedLayers(input_layer, output_layer);
}

int NeuralNetwork::setOptimizer(
  std::shared_ptr<ml::train::Optimizer> optimizer) {
  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  opt = std::static_pointer_cast<Optimizer>(optimizer);

  return ML_ERROR_NONE;
}

int NeuralNetwork::setDataBuffer(std::shared_ptr<DataBuffer> data_buffer) {
  this->data_buffer = data_buffer;

  return ML_ERROR_NONE;
}

int NeuralNetwork::getLayer(const char *name,
                            std::shared_ptr<ml::train::Layer> *layer) {
  *layer = std::static_pointer_cast<ml::train::Layer>(
    model_graph.getLayerNode(std::string(name)));
  return ML_ERROR_NONE;
}

/**
 * @brief     Set loss type for the neural network.
 */
int NeuralNetwork::setLoss(const std::string &loss_type) {
  this->loss_type = loss_type;
  return ML_ERROR_NONE;
}

void NeuralNetwork::printMetrics(std::ostream &out, unsigned int flags) {
  switch (flags) {
  case ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS:
    out << training.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_LOSS:
    out << validation.loss << std::endl;
    break;

  case ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY:
    out << validation.accuracy << std::endl;
    break;

  default:
    break;
  }
}

void NeuralNetwork::printPreset(std::ostream &out, unsigned int preset) {
  /** print neuralnet metrics */
  printMetrics(out, preset);
  if (preset > ML_TRAIN_SUMMARY_TENSOR)
    return;

  LayerV1::PrintPreset layer_preset = LayerV1::PrintPreset::PRINT_NONE;

  ///@todo match flags with preset
  unsigned int flags = PRINT_INST_INFO | PRINT_GRAPH_INFO | PRINT_PROP |
                       PRINT_OPTIMIZER | PRINT_METRIC;

  switch (preset) {
  case ML_TRAIN_SUMMARY_TENSOR:
    layer_preset = LayerV1::PrintPreset::PRINT_ALL;
    break;
  case ML_TRAIN_SUMMARY_LAYER:
    layer_preset = initialized ? LayerV1::PrintPreset::PRINT_SUMMARY
                               : LayerV1::PrintPreset::PRINT_SUMMARY_META;
    break;
  case ML_TRAIN_SUMMARY_MODEL:
    break;
  default:
    throw std::invalid_argument("given verbosity is invalid");
  }

  print(out, flags, layer_preset);
}

void NeuralNetwork::print(std::ostream &out, unsigned int flags,
                          LayerV1::PrintPreset layerPrintPreset) {
  if (flags & PRINT_INST_INFO) {
    out << "===================";
    printInstance(out, this);
  }

  if (flags & PRINT_GRAPH_INFO) {
    out << "graph contains " << model_graph.size() << " operation nodes\n";
    /// @todo print graph info
  }

  if (flags & PRINT_PROP) {
    /// @todo print neuralnet property
    /// @todo print mode (if it is eval or training)
  }

  if (flags & PRINT_OPTIMIZER) {
    /// @todo print optimizer (with print optimizer prop)
  }

  if (flags & PRINT_METRIC) {
    /// @todo print metric (currently it is done at printPreset as a
    /// workaround)
    /// @todo print loss function when it is not initialized. (if it is
    /// initialized, loss layer will be printed)
  }

  if (model_graph.empty()) {
    out << "model is empty!" << std::endl;
    return;
  }

  /** print layer properties */
  // TODO: get sorted layers if initialized
  // for (auto &layer : model_graph)
  // TODO: either support printPreset in LayerNode or use exportTo
  // layer->printPreset(out, layerPrintPreset);

  /// @todo Add status to check neuralnet has been run. #290
}

void NeuralNetwork::setSavePath(const std::string &path) {
  save_path = app_context.getWorkingPath(path);
  if (!isFileExist(save_path)) {
    ml_logw("[NeuralNetworks] save path does not exist, file will be newly "
            "created, path: %s",
            save_path.c_str());
  }
}

} /* namespace nntrainer */
