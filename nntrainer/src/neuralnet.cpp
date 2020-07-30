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

#include <array>
#include <assert.h>
#include <cmath>
#include <databuffer_file.h>
#include <databuffer_func.h>
#include <fstream>
#include <iniparser.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <sstream>
#include <stdio.h>

#define NN_INI_RETURN_STATUS()     \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      iniparser_freedict(ini);     \
      return status;               \
    }                              \
  } while (0)

#define VERIFY_SET_DIMENSION()                                     \
  do {                                                             \
    if (i != 0) {                                                  \
      if (def_init_dim == layers[i]->getInputDimension()) {        \
        layers[i]->setInputDimension(previous_dim);                \
      } else if (previous_dim != layers[i]->getInputDimension()) { \
        status = ML_ERROR_INVALID_PARAMETER;                       \
        ml_loge("Dimension mismatch between layers.");             \
        NN_RETURN_STATUS();                                        \
      }                                                            \
    }                                                              \
  } while (0)

#define CONV2D_DIM 2

namespace nntrainer {

/**
 * @brief     Check Existance of File
 * @param[in] filename file path to check
 * @retval    boolean true if exists
 */
static bool is_file_exist(std::string file_name) {
  std::ifstream infile(file_name);
  return infile.good();
}

NeuralNetwork::NeuralNetwork() :
  batch_size(1),
  epoch(1),
  loss(0.0f),
  cost(COST_UNKNOWN),
  weight_ini(WEIGHT_UNKNOWN),
  net_type(NET_UNKNOWN),
  data_buffer(NULL),
  continue_train(false),
  iter(0),
  initialized(false),
  def_name_count(0) {}

NeuralNetwork::NeuralNetwork(std::string config) : NeuralNetwork() {
  this->setConfig(config);
}

void NeuralNetwork::setConfig(std::string config) {
  if (!is_file_exist(config)) {
    std::stringstream ss;
    ss << "Cannot open model configuration file, filename : " << config;
    throw std::invalid_argument(ss.str().c_str());
  }
  this->config = config;
}

int NeuralNetwork::loadNetworkConfig(void *_ini) {
  dictionary *ini = static_cast<dictionary *>(_ini);
  char unknown[] = "Unknown";
  int status = ML_ERROR_NONE;

  /** Default to neural network model type */
  net_type = (nntrainer::NetType)parseType(
    iniparser_getstring(ini, "Network:Type", unknown), TOKEN_NET);
  epoch = iniparser_getint(ini, "Network:Epoch", epoch);
  cost = (CostType)parseType(iniparser_getstring(ini, "Network:Cost", unknown),
                             TOKEN_COST);
  model = iniparser_getstring(ini, "Network:Model", "model.bin");
  batch_size = iniparser_getint(ini, "Network:Minibatch", batch_size);

  /** Default to adam optimizer */
  status = opt.setType((OptType)parseType(
    iniparser_getstring(ini, "Network:Optimizer", "adam"), TOKEN_OPT));
  NN_INI_RETURN_STATUS();

  OptParam popt(opt.getType());
  popt.learning_rate =
    iniparser_getdouble(ini, "Network:Learning_rate", popt.learning_rate);
  popt.decay_steps =
    iniparser_getint(ini, "Network:Decay_steps", popt.decay_steps);
  popt.decay_rate =
    iniparser_getdouble(ini, "Network:Decay_rate", popt.decay_rate);
  popt.beta1 = iniparser_getdouble(ini, "Network:beta1", popt.beta1);
  popt.beta2 = iniparser_getdouble(ini, "Network:beta2", popt.beta2);
  popt.epsilon = iniparser_getdouble(ini, "Network:epsilon", popt.epsilon);

  status = opt.setOptParam(popt);
  NN_INI_RETURN_STATUS();

  return status;
}

/// @fixme: 370
int NeuralNetwork::loadDatasetConfig(void *_ini) {
  int status = ML_ERROR_NONE;

  dictionary *ini = static_cast<dictionary *>(_ini);

  if (iniparser_find_entry(ini, "DataSet:Tflite")) {
    ml_loge("Error: Tflite dataset is not yet implemented!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  data_buffer = std::make_shared<DataBufferFromDataFile>();
  std::shared_ptr<DataBufferFromDataFile> dbuffer =
    std::static_pointer_cast<DataBufferFromDataFile>(data_buffer);

  std::function<int(const char *, DataType, bool)> parse_and_set =
    [&](const char *key, DataType dt, bool required) -> int {
    const char *path = iniparser_getstring(ini, key, NULL);

    if (path == NULL) {
      return required ? ML_ERROR_INVALID_PARAMETER : ML_ERROR_NONE;
    }

    return dbuffer->setDataFile(path, dt);
  };

  status = parse_and_set("DataSet:TrainData", DATA_TRAIN, true);
  NN_INI_RETURN_STATUS();
  status = parse_and_set("DataSet:ValidData", DATA_VAL, false);
  NN_INI_RETURN_STATUS();
  status = parse_and_set("DataSet:TestData", DATA_TEST, false);
  NN_INI_RETURN_STATUS();
  status = parse_and_set("Dataset:LabelData", DATA_LABEL, true);
  NN_INI_RETURN_STATUS();

  /// fixme: #299
  status = data_buffer->setBufSize(
    iniparser_getint(ini, "DataSet:BufferSize", batch_size));
  NN_INI_RETURN_STATUS();

  return status;
}

int NeuralNetwork::loadFromConfig() {
  int status = ML_ERROR_NONE;
  std::string ini_file = config;
  int num_ini_sec = 0;
  dictionary *ini;
  const char network_str[] = "network";
  unsigned int network_len = strlen(network_str);
  const char dataset_str[] = "dataset";
  unsigned int dataset_len = strlen(dataset_str);
  const char unknown[] = "Unknown";
  unsigned int unknown_len = strlen(unknown);

  if (ini_file.empty()) {
    ml_loge("Error: Configuration File is not defined");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** Parse ini file */
  ini = iniparser_load(ini_file.c_str());
  if (ini == NULL) {
    ml_loge("Error: cannot parse file: %s\n", ini_file.c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** Get number of sections in the file */
  num_ini_sec = iniparser_getnsec(ini);
  if (num_ini_sec < 0) {
    ml_loge("Error: invalid number of sections.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (iniparser_find_entry(ini, "network") == 0) {
    ml_loge("there is no [network] section in given ini file");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** Get all the section names */
  ml_logi("==========================parsing ini... \n"
          "invalid properties does not cause error, rather be ignored \n"
          "not-allowed property for the layer throws error \n"
          "valid property with invalid value throws error as well");
  for (int idx = 0; idx < num_ini_sec; ++idx) {
    const char *sec_name = iniparser_getsecname(ini, idx);

    if (!sec_name) {
      ml_loge("Error: Unable to retrieve section names from ini.");
      status = ML_ERROR_INVALID_PARAMETER;
      NN_RETURN_STATUS();
    }

    if (strncasecmp(network_str, sec_name, network_len) == 0) {
      status = loadNetworkConfig((void *)ini);
      NN_RETURN_STATUS();
      continue;
    }

    if (strncasecmp(dataset_str, sec_name, dataset_len) == 0) {
      status = loadDatasetConfig((void *)ini);
      NN_RETURN_STATUS();
      continue;
    }

    /** Parse all the layers defined as sections in order */
    std::string layer_name(sec_name);

    std::string layer_type_str =
      iniparser_getstring(ini, (layer_name + ":Type").c_str(), unknown);
    LayerType layer_type = (LayerType)parseType(layer_type_str, TOKEN_LAYER);

    std::shared_ptr<Layer> layer;

    switch (layer_type) {
    case LAYER_IN:
      layer = std::make_shared<InputLayer>();
      break;
    case LAYER_CONV2D:
      layer = std::make_shared<Conv2DLayer>();
      break;
    case LAYER_POOLING2D:
      layer = std::make_shared<Pooling2DLayer>();
      break;
    case LAYER_FLATTEN:
      layer = std::make_shared<FlattenLayer>();
      break;
    case LAYER_FC:
      layer = std::make_shared<FullyConnectedLayer>();
      break;
    case LAYER_BN:
      layer = std::make_shared<BatchNormalizationLayer>();
      break;
    case LAYER_ACTIVATION:
      layer = std::make_shared<ActivationLayer>();
      break;
    case LAYER_UNKNOWN:
    default:
      ml_loge("Error: Unknown layer type");
      status = ML_ERROR_INVALID_PARAMETER;
      NN_INI_RETURN_STATUS();
    }

    unsigned int property_end =
      static_cast<unsigned int>(Layer::PropertyType::unknown);

    for (unsigned int i = 0; i < property_end; ++i) {
      std::string prop = propToStr(i);
      std::string value =
        iniparser_getstring(ini, (layer_name + ":" + prop).c_str(), unknown);

      /**! @todo: add following negative tc after #319
       * 1. layer has empty prop -> throw std::invalid_argument
       * 2. layer has not allowed property -> throw exception::invalid_property
       * 3. property value parse error -> throw std::invalid_argument
       */
      if (!strncmp(value.c_str(), unknown, unknown_len)) {
        continue;
      }

      if (value == "") {
        std::stringstream ss;
        ss << "property key " << prop << " has empty value. It is not allowed";
        throw std::invalid_argument(ss.str());
      }

      layer->setProperty(static_cast<Layer::PropertyType>(i), value);
    }

    status = layer->setName(layer_name);
    NN_INI_RETURN_STATUS();

    status = addLayer(layer);
    NN_INI_RETURN_STATUS();
  }

  /**< Additional validation and handling for the neural network */
  if (!data_buffer) {
    data_buffer = std::make_shared<DataBufferFromCallback>();
  }

  status = data_buffer->setMiniBatch(batch_size);
  NN_INI_RETURN_STATUS();

  if (layers.empty()) {
    ml_loge("there is no layer section in the ini file");
    return ML_ERROR_INVALID_PARAMETER;
  }

  iniparser_freedict(ini);
  return status;
}

int NeuralNetwork::initLossLayer() {
  int status = ML_ERROR_NONE;
  CostType updated_cost = cost;

  if (updated_cost == COST_ENTROPY) {
    if (layers.back()->getType() != LAYER_ACTIVATION) {
      ml_loge("Error: Cross Entropy need last layer to have softmax or sigmoid "
              "activation.");
      return ML_ERROR_NOT_SUPPORTED;
    }

    std::shared_ptr<ActivationLayer> act_layer =
      std::static_pointer_cast<ActivationLayer>(layers.back());
    layers.pop_back();

    switch (act_layer->getActivationType()) {
    case ACT_SIGMOID:
      updated_cost = COST_ENTROPY_SIGMOID;
      break;
    case ACT_SOFTMAX:
      updated_cost = COST_ENTROPY_SOFTMAX;
      break;
    default:
      ml_loge("Error: Cross Entropy not supported without softmax or sigmoid.");
      return ML_ERROR_NOT_SUPPORTED;
    }
  }

  std::shared_ptr<LossLayer> loss_layer = std::make_shared<LossLayer>();
  ensureName(loss_layer);

  loss_layer->setInputDimension(layers.back()->getOutputDimension());
  status = loss_layer->initialize(true);
  NN_RETURN_STATUS();

  status = layers.back()->setCost(updated_cost);
  NN_RETURN_STATUS();
  status = loss_layer->setCost(updated_cost);
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
    case PropertyType::batch_size: {
      status = setInt(batch_size, value);
      NN_RETURN_STATUS();
      for (unsigned int i = 0; i < layers.size(); ++i) {
        if (layers[i]->getTensorDim().batch() !=
            static_cast<unsigned int>(batch_size)) {
          ml_logw("Warning: Batch Size is changing!! : %d -> %d",
                  layers[i]->getTensorDim().batch(), batch_size);
          layers[i]->getTensorDim().batch(batch_size);
        }
      }
    } break;
    case PropertyType::cost:
    case PropertyType::loss: {
      cost = (CostType)parseType(value, TOKEN_COST);
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
      int e;
      status = setInt(e, value);
      NN_RETURN_STATUS();
      epoch = e;
    } break;
    case PropertyType::model_file: {
      model = value;
    } break;
    case PropertyType::continue_train: {
      bool cont_train;
      status = setBoolean(cont_train, value);
      NN_RETURN_STATUS();
      continue_train = cont_train;
      opt.setProperty({values[i]});
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
  bool last = false;
  TensorDim previous_dim, def_init_dim;

  /** Note: number of entries in layers will change. */
  for (unsigned int i = 0; i < layers.size(); ++i) {
    if (i == layers.size() - 1)
      last = true;

    VERIFY_SET_DIMENSION();

    switch (layers[i]->getType()) {
    case LAYER_IN: {
      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();
    } break;
    case LAYER_CONV2D: {
      std::shared_ptr<Conv2DLayer> conv2d_layer =
        std::static_pointer_cast<Conv2DLayer>(layers[i]);

      status = layers[i]->setCost(cost);
      NN_RETURN_STATUS();

      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();

      status = conv2d_layer->setOptimizer(opt);
      NN_RETURN_STATUS();

    } break;

    case LAYER_POOLING2D: {
      std::shared_ptr<Pooling2DLayer> pooling2d_layer =
        std::static_pointer_cast<Pooling2DLayer>(layers[i]);

      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();

    } break;

    case LAYER_FLATTEN: {
      std::shared_ptr<FlattenLayer> flatten_layer =
        std::static_pointer_cast<FlattenLayer>(layers[i]);

      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();
    } break;

    case LAYER_FC: {
      std::shared_ptr<FullyConnectedLayer> fc_layer =
        std::static_pointer_cast<FullyConnectedLayer>(layers[i]);

      status = layers[i]->setCost(cost);
      NN_RETURN_STATUS();

      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();

      status = fc_layer->setOptimizer(opt);
      NN_RETURN_STATUS();

    } break;
    case LAYER_BN:
      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();

      status = layers[i]->setOptimizer(opt);
      NN_RETURN_STATUS();

      if (i == 0) {
        ml_loge("Batch normalization layer cannot be first layer of network");
        status = ML_ERROR_INVALID_PARAMETER;
        NN_RETURN_STATUS();
      }
      layers[i - 1]->setBNfollow(true);
      break;
    default:
      break;
    }
    std::shared_ptr<Layer> last_layer = layers[i];
    status = initActivationLayer(last_layer->getActivationType(), i);
    NN_RETURN_STATUS();
    if (last_layer->getFlatten()) {
      status = initFlattenLayer(i);
      NN_RETURN_STATUS();
    }
    previous_dim = layers[i]->getOutputDimension();
  }

  /** Add the last layer as loss layer */
  status = initLossLayer();
  NN_RETURN_STATUS();

  initialized = true;
  return status;
}

/**
 * @brief     free layers
 */
void NeuralNetwork::finalize() {
  layers.erase(layers.begin(), layers.end());

  if (data_buffer) {
    data_buffer->clear();
  }
}

/**
 * @brief     forward propagation using layers object which has layer
 */
Tensor NeuralNetwork::forwarding(Tensor input, int &status) {
  Tensor X = input;
  /** Do not forward the loss layer, as label is not available */
  for (unsigned int i = 0; i < layers.size() - 1; i++) {
    X = layers[i]->forwarding(X, status);
    if (status != ML_ERROR_NONE)
      break;
  }
  return X;
}

/**
 * @brief     forward propagation using layers object which has layer
 */
Tensor NeuralNetwork::forwarding(Tensor input, Tensor output, int &status) {
  Tensor X = input;
  Tensor Y2 = output;

  X = forwarding(input, status);
  if (status != ML_ERROR_NONE)
    return X;

  X = std::static_pointer_cast<LossLayer>(layers[layers.size() - 1])
        ->forwarding(X, Y2, status);
  return X;
}

/**
 * @brief     back propagation
 *            Call backwarding function of layer in reverse order
 *            No need to call at first Input Layer (No data to be updated)
 */
int NeuralNetwork::backwarding(Tensor input, Tensor expected_output,
                               int iteration) {
  int status = ML_ERROR_NONE;
  Tensor Y2 = expected_output;
  Tensor X = input;
  Tensor Y = forwarding(X, Y2, status);
  if (status != ML_ERROR_NONE)
    return status;

  for (unsigned int i = layers.size() - 1; i > 0; i--) {
    Y2 = layers[i]->backwarding(Y2, iteration);
  }
  return status;
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
 * @brief     save model
 *            save Weight & Bias Data into file by calling save from layer
 *            save training parameters from the optimizer
 */
void NeuralNetwork::saveModel() {
  std::ofstream model_file(model, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->save(model_file);
  model_file.write((char *)&iter, sizeof(iter));
  model_file.close();
}

/**
 * @brief     read model
 *            read Weight & Bias Data into file by calling save from layer
 *            read training parameters from the optimizer if continuing train
 */
void NeuralNetwork::readModel() {
  if (!is_file_exist(model))
    return;
  std::ifstream model_file(model, std::ios::in | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->read(model_file);
  if (continue_train) {
    model_file.read((char *)&iter, sizeof(iter));
  }
  model_file.close();
  ml_logi("read modelfile: %s", model.c_str());
}

int NeuralNetwork::train() {
  std::vector<std::string> values;

  return train(values);
}

int NeuralNetwork::train(std::vector<std::string> values) {
  int status = ML_ERROR_NONE;

  if (data_buffer == nullptr) {
    ml_loge("Cannot initialize the model without the data buffer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = data_buffer->setMiniBatch(layers[0]->getInputDimension().batch());
  NN_RETURN_STATUS();

  status = data_buffer->setClassNum(
    layers[layers.size() - 1]->getOutputDimension().width());
  NN_RETURN_STATUS();

  status = data_buffer->setFeatureSize(layers[0]->getInputDimension());
  NN_RETURN_STATUS();

  status = data_buffer->init();
  NN_RETURN_STATUS();

  status = setTrainConfig(values);
  NN_RETURN_STATUS();

  return train_run();
}

/**
 * @brief     Run NeuralNetwork train with callback function by user
 */
int NeuralNetwork::train_run() {
  int status = ML_ERROR_NONE;

  float training_loss = 0.0f;
  for (unsigned int i = 0; i < epoch; ++i) {

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
        status =
          backwarding(nntrainer::Tensor(in), nntrainer::Tensor(label), iter++);
        if (status != ML_ERROR_NONE) {
          data_buffer->clear(nntrainer::BUF_TRAIN);
          ml_loge("Error: training error in #%d/%d.", i + 1, epoch);
          return status;
        }
        std::cout << "#" << i + 1 << "/" << epoch;
        data_buffer->displayProgress(count++, nntrainer::BUF_TRAIN, getLoss());
      } else {
        data_buffer->clear(nntrainer::BUF_TRAIN);
        break;
      }
    }

    saveModel();
    training_loss = getLoss();

    std::cout << "#" << i + 1 << "/" << epoch
              << " - Training Loss: " << training_loss;

    if (data_buffer->getValidation()[nntrainer::BUF_VAL]) {
      int right = 0;
      float valloss = 0.0f;
      unsigned int tcases = 0;

      status = data_buffer->run(nntrainer::BUF_VAL);
      if (status != ML_ERROR_NONE) {
        data_buffer->clear(BUF_VAL);
        return status;
      }

      while (true) {
        vec_4d in, label;
        if (data_buffer->getDataFromBuffer(nntrainer::BUF_VAL, in, label)) {
          for (int i = 0; i < batch_size; ++i) {
            nntrainer::Tensor X = nntrainer::Tensor({in[i]});
            nntrainer::Tensor Y2 = nntrainer::Tensor({label[i]});
            nntrainer::Tensor Y = forwarding(X, Y2, status);
            if (status != ML_ERROR_NONE) {
              ml_loge("Error: forwarding the network resulted in error.");
              return status;
            }

            if (Y.argmax() == Y2.argmax())
              right++;
            valloss += getLoss();
            tcases++;
          }
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
      valloss = valloss / (float)(tcases);
      std::cout << " >> [ Accuracy: " << right / (float)(tcases)*100.0f
                << "% - Validation Loss : " << valloss << " ] ";
    }
    std::cout << std::endl;
  }

  return status;
}

int NeuralNetwork::checkValidation() {
  int status = ML_ERROR_NONE;
  if (layers.empty()) {
    return ML_ERROR_INVALID_PARAMETER;
  } else {
    for (std::vector<std::shared_ptr<nntrainer::Layer>>::iterator layer =
           layers.begin();
         layer != layers.end(); ++layer) {
      status = (*layer)->checkValidation();
      if (status != ML_ERROR_NONE)
        return status;
    }
  }

  return status;
}

int NeuralNetwork::addLayer(std::shared_ptr<Layer> layer) {
  int status = ML_ERROR_NONE;

  LayerType type = layer->getType();
  if (type == LAYER_UNKNOWN)
    return ML_ERROR_INVALID_PARAMETER;

  if (initialized) {
    return ML_ERROR_NOT_SUPPORTED;
  }

  ensureName(layer);

  /** @todo This might be redundant. Remove this after testing */
  for (auto iter = layers.begin(); iter != layers.end(); ++iter) {
    if ((*iter)->getName() == layer->getName()) {
      ml_loge("Layer with name %s already exists in the model.",
              layer->getName().c_str());
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

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

    layer_names.insert(name);
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

std::shared_ptr<Layer>
NeuralNetwork::_make_act_layer(ActiType act, std::shared_ptr<Layer> prev) {
  if (layers.back()->getType() == LAYER_ACTIVATION) {
    /** User defined activation layer. Do not add another activation layer after
     * this */
    ml_loge("Error: double activation layers.");
    return nullptr;
  }

  if (act != ACT_UNKNOWN) {
    std::shared_ptr<ActivationLayer> act_layer =
      std::make_shared<ActivationLayer>();

    ensureName(act_layer, prev->getName());
    act_layer->setActivation(act);
    act_layer->setInputDimension(prev->getOutputDimension());
    act_layer->initialize(prev->getLast());
    return act_layer;
  }

  return nullptr;
}

int NeuralNetwork::initActivationLayer(ActiType act) {
  unsigned int position = layers.end() - layers.begin() - 1;
  return initActivationLayer(act, position);
}

int NeuralNetwork::initActivationLayer(ActiType act, unsigned int &position) {
  std::shared_ptr<Layer> l = _make_act_layer(act, layers[position]);
  if (l != nullptr) {
    layers.insert(layers.begin() + position + 1, l);
    position++;
    return ML_ERROR_NONE;
  }

  return ML_ERROR_INVALID_PARAMETER;
}

int NeuralNetwork::initFlattenLayer(unsigned int &position) {
  std::shared_ptr<FlattenLayer> flatten_layer =
    std::make_shared<FlattenLayer>();

  ensureName(flatten_layer, layers[position]->getName());
  flatten_layer->setInputDimension(layers[position]->getOutputDimension());
  flatten_layer->initialize(layers[position]->getLast());
  layers.insert(layers.begin() + position + 1, flatten_layer);
  position++;
  return ML_ERROR_NONE;
}

int NeuralNetwork::initFlattenLayer() {
  unsigned int position = layers.end() - layers.begin() - 1;
  return initFlattenLayer(position);
}

/**
 * @brief     Set cost type for the neural network.
 */
int NeuralNetwork::setCost(CostType cost) {
  if (cost == COST_UNKNOWN)
    return ML_ERROR_INVALID_PARAMETER;

  this->cost = cost;
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

void NeuralNetwork::print(std::ostream &out, unsigned int flags) {
  /// @todo print neuralnet property
  /// @todo print optimizer (with print optimizer prop)
  /// @todo print loss function when it is not initialized. (if it is
  /// initialized, loss layer will be printed)

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
  /// @todo print neuralnet metric after it is run
}

} /* namespace nntrainer */
