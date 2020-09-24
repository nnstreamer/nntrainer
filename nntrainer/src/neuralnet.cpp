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

  ModelLoader loader;
  NeuralNetwork tempNet(*this);
  int status = loader.loadFromConfig(config, tempNet);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  tempNet.loadedFromConfig = true;
  swap(tempNet, *this);

  return ML_ERROR_NONE;
}

int NeuralNetwork::initLossLayer() {
  int status = ML_ERROR_NONE;
  LossType updated_loss_type = loss_type;

  if (layers.empty()) {
    status = ML_ERROR_INVALID_PARAMETER;
    NN_RETURN_STATUS();
  }

  if (updated_loss_type == LossType::LOSS_ENTROPY) {
    if (layers.back()->getType() != LayerType::LAYER_ACTIVATION) {
      ml_loge("Error: Cross Entropy need last layer to have softmax or sigmoid "
              "activation.");
      return ML_ERROR_NOT_SUPPORTED;
    }

    std::shared_ptr<Layer> act_layer = layers.back();
    layers.pop_back();

    switch (act_layer->getActivationType()) {
    case ActivationType::ACT_SIGMOID:
      updated_loss_type = LossType::LOSS_ENTROPY_SIGMOID;
      break;
    case ActivationType::ACT_SOFTMAX:
      updated_loss_type = LossType::LOSS_ENTROPY_SOFTMAX;
      break;
    default:
      ml_loge("Error: Cross Entropy not supported without softmax or sigmoid.");
      return ML_ERROR_NOT_SUPPORTED;
    }
  }

  std::shared_ptr<LossLayer> loss_layer = std::make_shared<LossLayer>();
  ensureName(loss_layer);

  loss_layer->setInputDimension(layers.back()->getOutputDimension());
  status = loss_layer->initialize();
  NN_RETURN_STATUS();

  status = loss_layer->setLoss(updated_loss_type);
  NN_RETURN_STATUS();

  addLayer(loss_layer);
  return status;
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
      opt.setProperty({values[i]});
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

int NeuralNetwork::init() {
  int status = ML_ERROR_NONE;
  TensorDim previous_dim;

  status = isInitializable();
  NN_RETURN_STATUS();

  ml_logd("initiating neural network, layer size: %d",
          (unsigned int)layers.size());
  /** Note: number of entries in layers will change. */
  for (unsigned int i = 0; i < layers.size(); ++i) {
    bool first = i == 0;
    Layer &l = *layers[i];
    ml_logd("layer name: %s", l.getName().c_str());

    if (!first) {
      if (layers[i - 1]->getType() == LayerType::LAYER_ACTIVATION &&
          l.getType() == LayerType::LAYER_ACTIVATION) {
        ml_loge("double activation is not allowed");
        return ML_ERROR_INVALID_PARAMETER;
      }
      if (l.getInputDimension().isEmpty()) {
        l.setInputDimension(previous_dim);
      } else if (previous_dim != l.getInputDimension()) {
        ml_loge("Dimension mismatch between layers.");
        return ML_ERROR_INVALID_PARAMETER;
      }
    }

    status = layers[i]->initialize();

    switch (l.getType()) {
    case LayerType::LAYER_BN:
      /// fallthrough intended
    case LayerType::LAYER_CONV2D:
      /// fallthrough intended
    case LayerType::LAYER_FC:
      status = l.setOptimizer(opt);
      NN_RETURN_STATUS();
      break;
    default:
      break;
    }

    if (l.getType() != LayerType::LAYER_ACTIVATION) {
      status = realizeActivationType(l.getActivationType(), i);
      NN_RETURN_STATUS();
    }

    if (l.getFlatten()) {
      status = realizeFlattenType(i);
      NN_RETURN_STATUS();
    }

    previous_dim = l.getOutputDimension();
  }

  /** Add the last layer as loss layer */
  status = initLossLayer();
  NN_RETURN_STATUS();

  ml_logd("initialize successful, with layer size: %d", (int)layers.size());

  for (auto l : layers)
    ml_logd("layer name: %s", l->getName().c_str());

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
sharedConstTensor NeuralNetwork::forwarding(sharedConstTensor input) {
  sharedConstTensor X = input;
  /** Do not forward the loss layer, as label is not available */
  for (unsigned int i = 0; i < layers.size() - 1; i++) {
    X = layers[i]->forwarding(X);
  }

  return X;
}

/**
 * @brief     forward propagation using layers object which has layer
 */
sharedConstTensor NeuralNetwork::forwarding(sharedConstTensor input,
                                            sharedConstTensor label) {
  sharedConstTensor X;

  if (input->getDim().batch() > batch_size)
    throw std::logic_error("Error: mismatch in batchsize for data and model.");

  X = forwarding(input);
  X = std::static_pointer_cast<LossLayer>(layers[layers.size() - 1])
        ->forwarding(X, label);

  return X;
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
void NeuralNetwork::backwarding(sharedConstTensor input,
                                sharedConstTensor label, int iteration) {

  if (layers.empty() || layers.back()->getType() != LayerType::LAYER_LOSS) {
    throw std::invalid_argument("last layer is not loss layer");
  }

  forwarding(input, label);

  sharedConstTensor output = label;
  for (unsigned int i = layers.size() - 1; i > 0; i--)
    output = layers[i]->backwarding(output, iteration);
}

float NeuralNetwork::getLoss() {
  loss = 0.0f;
  for (unsigned int i = 0; i < layers.size(); i++) {
    loss += layers[i]->getLoss();
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
  std::ofstream model_file(save_path, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->save(model_file);
  model_file.write((char *)&iter, sizeof(iter));
  model_file.close();
}

/**
 * @brief     read model from file
 *            read Weight & Bias Data into file by calling save from layer
 *            read training parameters from the optimizer if continuing train
 */
void NeuralNetwork::readModel() {
  if (!isFileExist(save_path))
    return;
  std::ifstream model_file(save_path, std::ios::in | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->read(model_file);
  if (continue_train) {
    model_file.read((char *)&iter, sizeof(iter));
  }
  model_file.close();
  ml_logi("read modelfile: %s", save_path.c_str());
}

void NeuralNetwork::setBatchSize(unsigned int batch) {
  batch_size = batch;
  for (auto const &layer : layers)
    layer->setBatch(batch_size);

  if (data_buffer && data_buffer->setBatchSize(batch_size) != ML_ERROR_NONE)
    throw std::invalid_argument("Error setting batchsize for the dataset");
}

sharedConstTensor NeuralNetwork::inference(const Tensor X) {
  if (batch_size != X.batch()) {
    /**
     * Note that inference resets batch_size of the previous train configuration
     * Next train must set its batch_size if inference is run with this model.
     */
    setBatchSize(X.batch());
  }

  sharedConstTensor out;
  try {
    out = forwarding(MAKE_SHARED_TENSOR(X));
    /** Forward loss layer without label as well */
    out = std::static_pointer_cast<LossLayer>(layers[layers.size() - 1])
            ->forwarding(out);
  } catch (...) {
    ml_loge("Failed to inference Model");
    return nullptr;
  }
  return out;
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

  /** Setup data buffer properties */
  status =
    data_buffer->setClassNum(layers.back()->getOutputDimension().width());
  NN_RETURN_STATUS();

  status = data_buffer->setFeatureSize(layers[0]->getInputDimension());
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

  for (unsigned int epoch_idx = 1; epoch_idx <= epochs; ++epoch_idx) {
    training.loss = 0.0f;
    status = data_buffer->run(nntrainer::BUF_TRAIN);
    if (status != ML_ERROR_NONE) {
      data_buffer->clear(BUF_TRAIN);
      return status;
    }

    if (data_buffer->getValidation()[nntrainer::BUF_TEST]) {
      status = data_buffer->run(nntrainer::BUF_TEST);
      if (status != ML_ERROR_NONE) {
        data_buffer->clear(BUF_TEST);
        return status;
      }
    }

    int count = 0;

    while (true) {
      vec_4d in, label;
      if (data_buffer->getDataFromBuffer(nntrainer::BUF_TRAIN, in, label)) {
        try {
          backwarding(MAKE_SHARED_TENSOR(in), MAKE_SHARED_TENSOR(label),
                      iter++);
        } catch (...) {
          data_buffer->clear(nntrainer::BUF_TRAIN);
          ml_loge("Error: training error in #%d/%d.", epoch_idx, epochs);
          std::rethrow_exception(std::current_exception());
        }
        std::cout << "#" << epoch_idx << "/" << epochs;
        data_buffer->displayProgress(count++, nntrainer::BUF_TRAIN, getLoss());
        training.loss += getLoss();
      } else {
        data_buffer->clear(nntrainer::BUF_TRAIN);
        break;
      }
    }

    if (count == 0)
      throw std::runtime_error("No training data");

    training.loss /= count;
    saveModel();

    std::cout << "#" << epoch_idx << "/" << epochs
              << " - Training Loss: " << training.loss;

    if (data_buffer->getValidation()[nntrainer::BUF_VAL]) {
      int right = 0;
      validation.loss = 0.0f;
      unsigned int tcases = 0;

      status = data_buffer->run(nntrainer::BUF_VAL);
      if (status != ML_ERROR_NONE) {
        data_buffer->clear(BUF_VAL);
        return status;
      }

      while (true) {
        vec_4d in, label;
        if (data_buffer->getDataFromBuffer(nntrainer::BUF_VAL, in, label)) {
          sharedTensor X = MAKE_SHARED_TENSOR(Tensor({in}));
          sharedTensor Y2 = MAKE_SHARED_TENSOR(Tensor({label}));
          sharedConstTensor Y = forwarding(X, Y2);
          auto model_out = Y->argmax();
          auto label_out = Y2->argmax();
          for (unsigned int b = 0; b < batch_size; b++) {
            if (model_out[b] == label_out[b])
              right++;
          }
          validation.loss += getLoss();
          tcases++;
        } else {
          data_buffer->clear(nntrainer::BUF_VAL);
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

int NeuralNetwork::isInitializable() {
  if (layers.empty()) {
    ml_loge("Layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  Layer &l = *layers[0];

  /** Dimension of first layer must be known */
  if (l.getInputDimension().isEmpty()) {
    ml_loge("InputDimension of first layer is not set");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** First layer cannot be activation, batch normalization or loss */
  switch (l.getType()) {
  case LayerType::LAYER_ACTIVATION:
    /// fallthrough intended
  case LayerType::LAYER_BN:
    /// fallthrough intended
  case LayerType::LAYER_LOSS:
    /// fallthrough intended
    ml_loge("%s cannot be the first layer, type: %d", l.getName().c_str(),
            static_cast<std::underlying_type<LayerType>::type>(l.getType()));
    return ML_ERROR_INVALID_PARAMETER;
  default:
    /// normal case
    break;
  }

  return ML_ERROR_NONE;
}

int NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
  int status = ML_ERROR_NONE;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  ensureName(layer);

  /** Validate the layer to be added */
  status = layer->checkValidation();
  if (status != ML_ERROR_NONE) {
    ml_loge("layer(%s) validation failed.", layer->getName().c_str());
    return status;
  }

  /** Ensure that the layer name is unique */
  auto insert_result = layer_names.insert(layer->getName());
  if (!insert_result.second) {
    std::stringstream ss;
    ss << "Layer with name " << layer->getName() << " already exists";
    throw std::invalid_argument(ss.str());
  }

  /** Insert the layer to the graph */
  layers.push_back(layer);

  return status;
}

int NeuralNetwork::setOptimizer(std::shared_ptr<Optimizer> optimizer) {

  if (optimizer->getType() == OptType::unknown)
    return ML_ERROR_INVALID_PARAMETER;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  opt = *optimizer.get();

  return ML_ERROR_NONE;
}

int NeuralNetwork::setDataBuffer(std::shared_ptr<DataBuffer> data_buffer) {
  this->data_buffer = data_buffer;

  return ML_ERROR_NONE;
}

void NeuralNetwork::ensureName(std::shared_ptr<Layer> layer,
                               std::string prefix) {
  if (layer->getName().empty()) {
    std::set<std::string>::iterator iter;
    std::string name;

    do {
      name = prefix + layer->getBaseName() + std::to_string(def_name_count++);
      iter = layer_names.find(name);
    } while (iter != layer_names.end());

    layer->setName(name);
  }
}

int NeuralNetwork::getLayer(const char *name, std::shared_ptr<Layer> *layer) {
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

int NeuralNetwork::realizeActivationType(const ActivationType act) {
  unsigned int position = layers.end() - layers.begin() - 1;
  return realizeActivationType(act, position);
}

int NeuralNetwork::realizeActivationType(const ActivationType act,
                                         const unsigned int position) {
  if (act == ActivationType::ACT_NONE) {
    /// ActivationType::ACT_NONE does not need realization
    return ML_ERROR_NONE;
  }

  if (layers.empty()) {
    ml_loge("layer is empty");
    return ML_ERROR_INVALID_PARAMETER;
  }

  Layer &current = *layers[position];
  if (current.getType() == LayerType::LAYER_ACTIVATION) {
    ml_loge("It is not allowed to realize ativation layer, possibly layer is "
            "added right after activation");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (act == ActivationType::ACT_UNKNOWN) {
    ml_loge("cannot realize unknown activation type");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<ActivationLayer> act_layer =
    std::make_shared<ActivationLayer>();
  ensureName(act_layer, current.getName());
  act_layer->setActivation(act);

  layers.insert(layers.begin() + position + 1, act_layer);
  return ML_ERROR_NONE;
}

int NeuralNetwork::realizeFlattenType(const unsigned int position) {
  if (layers.empty()) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  Layer &current = *layers[position];
  if (current.getType() == LayerType::LAYER_FLATTEN) {
    ml_loge(
      "It is not allowed to realize flatten layer, possibly flatten layer is "
      "added right after flatten");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<FlattenLayer> flatten_layer =
    std::make_shared<FlattenLayer>();

  ensureName(flatten_layer, current.getName());
  layers.insert(layers.begin() + position + 1, flatten_layer);
  return ML_ERROR_NONE;
}

int NeuralNetwork::realizeFlattenType() {
  unsigned int position = layers.end() - layers.begin() - 1;
  return realizeFlattenType(position);
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

static unsigned int getLayerFlag(ml_train_summary_type_e verbosity,
                                 bool initialized = false) {
  unsigned int flag = 0;

  switch (verbosity) {
  case ML_TRAIN_SUMMARY_TENSOR:
    flag |= LayerPrintOption::PRINT_WEIGHTS;
    /// no break intended

  case ML_TRAIN_SUMMARY_LAYER:
    if (!initialized)
      flag |= LayerPrintOption::PRINT_PROP_META;
    else
      flag |= LayerPrintOption::PRINT_METRIC;
    flag |= LayerPrintOption::PRINT_PROP;
    /// no break intended

  case ML_TRAIN_SUMMARY_MODEL:
    flag |=
      LayerPrintOption::PRINT_INST_INFO | LayerPrintOption::PRINT_SHAPE_INFO;
    break;

  default:
    throw std::invalid_argument("given verbosity is invalid");
  }

  return flag;
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

void NeuralNetwork::print(std::ostream &out, unsigned int flags) {
  /// @todo print neuralnet property
  /// @todo print optimizer (with print optimizer prop)
  /// @todo print loss function when it is not initialized. (if it is
  /// initialized, loss layer will be printed)

  /** print neuralnet metrics */
  printMetrics(out, flags);
  if (flags > ML_TRAIN_SUMMARY_TENSOR)
    return;

  /** print layer properties */
  if (layers.empty()) {
    out << "model is empty!" << std::endl;
    return;
  }
  unsigned int layerFlag =
    getLayerFlag((ml_train_summary_type_e)flags, initialized);

  for (auto &layer : layers) {
    layer->print(out, layerFlag);
  }

  /// @todo Add status to check neuralnet has been run. #290
}

} /* namespace nntrainer */
