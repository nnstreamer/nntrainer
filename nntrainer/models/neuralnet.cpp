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
#include <fstream>
#include <sstream>

#include <databuffer_file.h>
#include <databuffer_func.h>
#include <model_loader.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
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
      loss_type = (LossType)parseType(value, TOKEN_LOSS);
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
  int status = ML_ERROR_NONE;

  status = isCompilable();
  NN_RETURN_STATUS();

  ml_logd("Compile model");

  status = model_graph.setGraphNode(layers, loss_type);
  NN_RETURN_STATUS();

  status = model_graph.setEdge();
  NN_RETURN_STATUS();

  model_graph.topologicalSort();

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

  unsigned int n_layers = (unsigned int)model_graph.Sorted.size();

  ml_logd("initializing neural network, layer size: %d", n_layers);

  model_graph.setNumNetBufferSize();

  for (unsigned int idx = 0; idx < n_layers; ++idx) {
    bool first = idx == 0;
    Layer &l = *model_graph.getSortedLayerNode(idx).layer;
    ml_logd("layer name : %s", l.getName().c_str());
    const std::string &cur_type = l.getType();

    if (!first) {
      if (istrequal(model_graph.getSortedLayerNode(idx - 1).layer->getType(),
                    ActivationLayer::type) &&
          istrequal(cur_type, ActivationLayer::type)) {
        ml_loge("double activation is not allowed");
        return ML_ERROR_INVALID_PARAMETER;
      }

      for (unsigned int i = 0; i < l.input_layers.size(); ++i) {
        std::shared_ptr<NetBuffers> n_buffer = std::make_unique<NetBuffers>();
        Layer &in_layer = *model_graph.getLayerNode(l.input_layers[i]).layer;

        unsigned int location = 0;
        for (unsigned int j = 0; j < in_layer.output_layers.size(); ++j) {
          if (in_layer.output_layers[j] == l.getName()) {
            location = j;
            break;
          }
        }

        l.setInputDimension(in_layer.getOutputDimension()[location], i);

        l.net_input[i] = n_buffer;

        model_graph.getLayerNode(l.input_layers[i])
          .layer->net_hidden[location] = n_buffer;
      }
    } else {
      for (unsigned int i = 0; i < l.input_layers.size(); ++i) {
        std::shared_ptr<NetBuffers> n_buffer = std::make_unique<NetBuffers>();
        l.net_input[i] = n_buffer;
      }
    }

    status = l.initialize(manager);
    NN_RETURN_STATUS();

    if (istrequal(cur_type, BatchNormalizationLayer::type) ||
        istrequal(cur_type, Conv2DLayer::type) ||
        istrequal(cur_type, FullyConnectedLayer::type)) {
      status = l.setOptimizer(opt);
      NN_RETURN_STATUS();
    }
  }

  for (unsigned int i = 0; i < model_graph.Sorted.back().layer->num_outputs;
       ++i) {
    std::shared_ptr<NetBuffers> last_hidden_buffer =
      std::make_unique<NetBuffers>();
    model_graph.Sorted.back().layer->net_hidden[i] = last_hidden_buffer;
  }

  setBatchSize(batch_size);

  manager.initialize();

  initialized = true;
  return status;
}

/**
 * @brief     free layers
 */
NeuralNetwork::~NeuralNetwork() {
  layers.erase(layers.begin(), layers.end());

  if (data_buffer) {
    data_buffer->clear();
  }
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(sharedConstTensors input) {
  sharedConstTensors X = model_graph.forwarding(input);
  return X;
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensors NeuralNetwork::forwarding(sharedConstTensors input,
                                             sharedConstTensors label) {
  sharedConstTensors X;
  if (input[0]->getDim().batch() > batch_size)
    throw std::logic_error("Error: mismatch in batchsize for data and model.");

  X = forwarding(input);
  X = std::static_pointer_cast<LossLayer>(model_graph.Sorted.back().layer)
        ->forwarding(X, label);

  return X;
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(sharedConstTensors input,
                                sharedConstTensors label, int iteration) {

  if (model_graph.Sorted.empty() ||
      !istrequal(model_graph.Sorted.back().layer->getType(), LossLayer::type)) {
    throw std::invalid_argument("last layer is not loss layer");
  }

  forwarding(input, label);

  sharedConstTensors output = label;

  model_graph.backwarding(output, iteration);
}

float NeuralNetwork::getLoss() {
  loss = 0.0f;

  for (unsigned int i = 0; i < model_graph.Sorted.size(); i++) {
    loss += model_graph.Sorted[i].layer->getLoss();
  }
  return loss;
}

void NeuralNetwork::setLoss(float l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    batch_size = from.batch_size;
    loss = from.loss;
    opt = from.opt;

    for (unsigned int i = 0; i < layers.size(); i++)
      layers[i]->copy(from.layers[i]);
  }
  return *this;
}

/**
 * @brief     save model to file
 *            save Weight & Bias Data into file by calling save from layer
 *            save training parameters from the optimizer
 */
void NeuralNetwork::saveModel() {
  if (save_path == std::string()) {
    return;
  }

  std::ofstream model_file(save_path, std::ios::out | std::ios::binary);

  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->save(model_file);
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
  if (save_path == std::string()) {
    return;
  }

  if (!isFileExist(save_path)) {
    ml_logd("skipping reading model, path is not valid: %s", save_path.c_str());
    return;
  }

  NeuralNetwork tmp(*this);

  std::ifstream model_file(save_path, std::ios::in | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    tmp.layers[i]->read(model_file);
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

  if (data_buffer && data_buffer->setBatchSize(batch_size) != ML_ERROR_NONE)
    throw std::invalid_argument("Error setting batchsize for the dataset");
}

sharedConstTensors NeuralNetwork::inference(sharedConstTensors X) {
  if (batch_size != X[0]->batch()) {
    /**
     * Note that inference resets batch_size of the previous train configuration
     * Next train must set its batch_size if inference is run with this model.
     */
    setBatchSize(X[0]->batch());
  }

  assignMem();

  sharedConstTensors out;
  try {
#ifdef PROFILE
    profiler.start(Profiler::FORWARD);
#endif
    forwarding(X);
#ifdef PROFILE
    profiler.end(Profiler::FORWARD);
#endif
    /** Forward loss layer without label as well */
    std::static_pointer_cast<LossLayer>(model_graph.Sorted.back().layer)
      ->forwarding();
  } catch (...) {
    ml_loge("Failed to inference Model");
    return out;
  }

  for (unsigned int i = 0;
       i < model_graph.Sorted[model_graph.Sorted.size() - 1].layer->num_outputs;
       ++i) {
    out.push_back(
      MAKE_SHARED_TENSOR(model_graph.Sorted[model_graph.Sorted.size() - 1]
                           .layer->net_hidden[i]
                           ->var));
  }
  return out;
}

int NeuralNetwork::assignMem() {
  int status = ML_ERROR_NONE;
  unsigned int n_layers = (unsigned int)model_graph.Sorted.size();

  for (unsigned int idx = 0; idx < n_layers; ++idx) {
    bool first = idx == 0;
    Layer &l = *model_graph.getSortedLayerNode(idx).layer;

    if (!first) {
      for (unsigned int i = 0; i < l.input_layers.size(); ++i) {

        l.net_input[i]->var = Tensor(l.getInputDimension()[i]);
      }
    }
  }

  for (unsigned int i = 0; i < model_graph.Sorted.back().layer->num_outputs;
       ++i) {
    model_graph.Sorted.back().layer->net_hidden[i]->var =
      Tensor(model_graph.Sorted.back().layer->getOutputDimension()[i]);
  }

  return status;
}

int NeuralNetwork::train(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  if (data_buffer == nullptr) {
    ml_loge("Cannot initialize the model without the data buffer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = setTrainConfig(values);
  NN_RETURN_STATUS();

  /** set batch size just before training */
  setBatchSize(batch_size);

  status = assignMem();
  NN_RETURN_STATUS();

  /** Setup data buffer properties */
  status = data_buffer->setClassNum(getOutputDimension()[0].width());
  NN_RETURN_STATUS();

  status = data_buffer->setFeatureSize(getInputDimension()[0]);
  NN_RETURN_STATUS();

  status = data_buffer->init();
  NN_RETURN_STATUS();

  return train_run();
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

  sharedTensor in = MAKE_SHARED_TENSOR(getInputDimension()[0]);
  sharedTensor label = MAKE_SHARED_TENSOR(getOutputDimension()[0]);

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
                                         in->getData(), label->getData())) {
        try {
          backwarding({in}, {label}, iter++);
        } catch (...) {
          data_buffer->clear(nntrainer::BufferType::BUF_TRAIN);
          ml_loge("Error: training error in #%d/%d.", epoch_idx, epochs);
          std::rethrow_exception(std::current_exception());
        }
        std::cout << "#" << epoch_idx << "/" << epochs;
        data_buffer->displayProgress(count++, nntrainer::BufferType::BUF_TRAIN,
                                     getLoss());
        training.loss += getLoss();
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
                                           in->getData(), label->getData())) {
          sharedConstTensors Y = forwarding({in}, {label});
          auto model_out = Y[0]->argmax();
          auto label_out = label->argmax();
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

int NeuralNetwork::isCompilable() {
  if (layers.empty()) {
    ml_loge("Layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  Layer &l = *layers[0];

  /** Dimension of first layer must be known */
  if (l.getInputDimension().size() == 0) {
    ml_loge("InputDimension of first layer is not set");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** First layer cannot be activation, batch normalization or loss */
  const std::string &type = l.getType();
  if (istrequal(type, ActivationLayer::type) ||
      istrequal(type, BatchNormalizationLayer::type) ||
      istrequal(type, LossLayer::type)) {
    ml_loge("%s cannot be the first layer, type: %s", l.getName().c_str(),
            type.c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

int NeuralNetwork::addLayer(NodeType layer) {
  int status = ML_ERROR_NONE;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  /** Ensure that the layer has a name and is unique */
  model_graph.ensureName(layer);

  /** Validate the layer to be added */
  status = layer->checkValidation();
  if (status != ML_ERROR_NONE) {
    ml_loge("layer(%s) validation failed.", layer->getName().c_str());
    return status;
  }

  /** Insert the layer to the graph */
  layers.push_back(layer);

  return status;
}

int NeuralNetwork::extendGraph(GraphType graph, std::string prefix) {
  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  if (graph.size() == 0)
    return ML_ERROR_NONE;

  for (unsigned int i = 0; i < graph[0]->input_layers.size(); ++i) {
    if (sub_in_out.find(graph[0]->input_layers[i]) != sub_in_out.end()) {
      graph[0]->input_layers[i] = sub_in_out[graph[0]->input_layers[i]];
    }
  }

  /** Insert the layer to the graph */
  for (auto layer : graph) {
    /**
     * Add prefix to the existing layer name,
     * and ensure it is unique in this new graph
     */
    std::string org_name = prefix + layer->getName();
    model_graph.ensureName(layer, prefix, true);
    sub_in_out.insert(std::make_pair(org_name, layer->getName()));

    for (unsigned int i = 0; i < layer->input_layers.size(); ++i) {
      if (sub_in_out.find(prefix + layer->input_layers[i]) !=
          sub_in_out.end()) {
        layer->input_layers[i] = sub_in_out[prefix + layer->input_layers[i]];
      }
    }

    layers.push_back(layer);
  }

  sub_in_out.insert(std::make_pair(prefix, layers.back()->getName()));

  return ML_ERROR_NONE;
}

std::vector<std::shared_ptr<Layer>>
NeuralNetwork::getGraph(const std::string &input_layer,
                        const std::string &output_layer) {
  /** count layers after output layer */
  unsigned int num_layers_remove_end = 0;
  if (!output_layer.empty()) {
    for (auto iter = layers.rbegin(); iter != layers.rend(); iter++) {
      if ((*iter)->getName() != output_layer)
        num_layers_remove_end++;
      else
        break;
    }
  }

  if (num_layers_remove_end == layers.size())
    return {};

  /** count layers before input layer */
  unsigned int num_layers_remove_start = 0;
  if (!input_layer.empty()) {
    for (auto iter = layers.begin();
         iter != layers.end() - num_layers_remove_end; iter++) {
      if ((*iter)->getName() != input_layer)
        num_layers_remove_start++;
      else
        break;
    }
  }

  /** copy the graph and return */
  GraphType ret;
  std::copy(layers.begin() + num_layers_remove_start,
            layers.end() - num_layers_remove_end, std::back_inserter(ret));

  return ret;
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
  std::shared_ptr<Layer> layer_;
  int ret = getLayer(name, &layer_);
  if (ret == ML_ERROR_NONE)
    *layer = layer_;
  return ret;
}

int NeuralNetwork::getLayer(const char *name, NodeType *layer) {
  int status = ML_ERROR_INVALID_PARAMETER;
  std::string name_str(name);

  for (auto iter = layers.begin(); iter != layers.end(); ++iter) {
    if ((*iter)->getName() == name_str) {
      *layer = *iter;
      return ML_ERROR_NONE;
    }
  }

  return status;
}

/**
 * @brief     Set loss type for the neural network.
 */
int NeuralNetwork::setLoss(LossType loss_type) {
  if (loss_type == LossType::LOSS_UNKNOWN)
    return ML_ERROR_INVALID_PARAMETER;

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

  Layer::PrintPreset layer_preset = Layer::PrintPreset::PRINT_NONE;

  ///@todo match flags with preset
  unsigned int flags = PRINT_INST_INFO | PRINT_GRAPH_INFO | PRINT_PROP |
                       PRINT_OPTIMIZER | PRINT_METRIC;

  switch (preset) {
  case ML_TRAIN_SUMMARY_TENSOR:
    layer_preset = Layer::PrintPreset::PRINT_ALL;
    break;
  case ML_TRAIN_SUMMARY_LAYER:
    layer_preset = initialized ? Layer::PrintPreset::PRINT_SUMMARY
                               : Layer::PrintPreset::PRINT_SUMMARY_META;
    break;
  case ML_TRAIN_SUMMARY_MODEL:
    break;
  default:
    throw std::invalid_argument("given verbosity is invalid");
  }

  print(out, flags, layer_preset);
}

void NeuralNetwork::print(std::ostream &out, unsigned int flags,
                          Layer::PrintPreset layerPrintPreset) {
  if (flags & PRINT_INST_INFO) {
    out << "===================";
    printInstance(out, this);
  }

  if (flags & PRINT_GRAPH_INFO) {
    out << "graph contains " << layers.size() << " operation nodes\n";
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

  if (layers.empty()) {
    out << "model is empty!" << std::endl;
    return;
  }

  /** print layer properties */
  for (auto &layer : layers)
    layer->printPreset(out, layerPrintPreset);

  /// @todo Add status to check neuralnet has been run. #290
}

void NeuralNetwork::setSavePath(const std::string &path) {
  save_path = app_context.getWorkingPath(path);
}

} /* namespace nntrainer */
