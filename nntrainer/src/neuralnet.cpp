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

static int setWeightDecay(dictionary *ini, std::string layer_name,
                          WeightDecayParam &weight_decay) {
  char unknown[] = "Unknown";
  int status = ML_ERROR_NONE;
  weight_decay.type = (WeightDecayType)parseType(
    iniparser_getstring(ini, (layer_name + ":Weight_Decay").c_str(), unknown),
    TOKEN_WEIGHT_DECAY);

  weight_decay.lambda = 0.0;
  if (weight_decay.type == WeightDecayType::l2norm) {
    weight_decay.lambda = iniparser_getdouble(
      ini, (layer_name + ":Weight_Decay_Lambda").c_str(), 0.0);
  }
  return status;
}

/**
 * @brief     Parsing Layer Name
 * @param[in] string layer name
 * @retval    vector stored layer name
 */
std::vector<std::string> parseLayerName(std::string ll) {
  std::vector<std::string> ret;
  std::istringstream ss(ll);
  do {
    std::string word;
    ss >> word;
    if (word.compare("") != 0)
      ret.push_back(word);
  } while (ss);

  return ret;
}

NeuralNetwork::NeuralNetwork() : NeuralNetwork("") {}

NeuralNetwork::NeuralNetwork(std::string config) : batch_size(0),
  learning_rate(0.0),
  decay_rate(0.0),
  decay_steps(0.0),
  epoch(0),
  loss(0.0),
  cost(COST_UNKNOWN),
  weight_ini(WEIGHT_UNKNOWN),
  net_type(NET_UNKNOWN),
  data_buffer(NULL),
  continue_train(false),
  iter(0),
  initialized(false) {
  this->setConfig(config);
}

int NeuralNetwork::setConfig(std::string config) {
  int status = ML_ERROR_NONE;
  if (!is_file_exist(config)) {
    ml_loge("Error: Cannot open model configuration file");
    return ML_ERROR_INVALID_PARAMETER;
  }

  this->config = config;

  return status;
}

int NeuralNetwork::init() {
  int status = ML_ERROR_NONE;
  bool b_zero;
  std::string l_type;
  LayerType t;
  std::string ini_file = config;

  if (ini_file.empty()) {
    ml_loge("Error: Configuration File is not defined");
    return ML_ERROR_INVALID_PARAMETER;
  }

  dictionary *ini = iniparser_load(ini_file.c_str());
  std::vector<int> hidden_size;
  OptParam popt;

  char unknown[] = "Unknown";
  char model_name[] = "model.bin";

  if (ini == NULL) {
    ml_loge("Error: cannot parse file: %s\n", ini_file.c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  net_type = (nntrainer::NetType)parseType(
    iniparser_getstring(ini, "Network:Type", unknown), TOKEN_NET);
  std::vector<std::string> layers_name =
    parseLayerName(iniparser_getstring(ini, "Network:Layers", ""));
  if (!layers_name.size()) {
    ml_loge("Error: There is no layer");
    iniparser_freedict(ini);
    return ML_ERROR_INVALID_PARAMETER;
  }

  learning_rate = iniparser_getdouble(ini, "Network:Learning_rate", 0.0);
  decay_rate = iniparser_getdouble(ini, "Network:Decay_rate", 0.0);
  decay_steps = iniparser_getint(ini, "Network:Decay_steps", -1);

  popt.learning_rate = learning_rate;
  popt.decay_steps = decay_steps;
  popt.decay_rate = decay_rate;
  epoch = iniparser_getint(ini, "Network:Epoch", 100);
  status = opt.setType((OptType)parseType(
    iniparser_getstring(ini, "Network:Optimizer", unknown), TOKEN_OPT));
  NN_INI_RETURN_STATUS();

  cost = (CostType)parseType(iniparser_getstring(ini, "Network:Cost", unknown),
                             TOKEN_COST);
  model = iniparser_getstring(ini, "Network:Model", model_name);
  batch_size = iniparser_getint(ini, "Network:Minibatch", 1);

  popt.beta1 = iniparser_getdouble(ini, "Network:beta1", 0.0);
  popt.beta2 = iniparser_getdouble(ini, "Network:beta2", 0.0);
  popt.epsilon = iniparser_getdouble(ini, "Network:epsilon", 0.0);

  status = opt.setOptParam(popt);
  NN_INI_RETURN_STATUS();

  for (unsigned int i = 0; i < layers_name.size(); i++)
    ml_logi("%s", layers_name[i].c_str());

  loss = 100000.0;

  if (iniparser_find_entry(ini, "DataSet:TrainData")) {
    data_buffer = std::make_shared<DataBufferFromDataFile>();

    std::shared_ptr<DataBufferFromDataFile> dbuffer =
      std::static_pointer_cast<DataBufferFromDataFile>(data_buffer);

    status = dbuffer->setDataFile(
      iniparser_getstring(ini, "DataSet:TrainData", ""), DATA_TRAIN);
    NN_INI_RETURN_STATUS();
    status = dbuffer->setDataFile(
      iniparser_getstring(ini, "DataSet:ValidData", ""), DATA_VAL);
    NN_INI_RETURN_STATUS();
    status = dbuffer->setDataFile(
      iniparser_getstring(ini, "DataSet:TestData", ""), DATA_TEST);
    NN_INI_RETURN_STATUS();
    status = dbuffer->setDataFile(
      iniparser_getstring(ini, "DataSet:LabelData", ""), DATA_LABEL);
    NN_INI_RETURN_STATUS();

  } else if (iniparser_find_entry(ini, "DataSet:Tflite")) {
    ml_loge("Error: Not yet implemented!");
    return ML_ERROR_INVALID_PARAMETER;
  } else {
    data_buffer = std::make_shared<DataBufferFromCallback>();
  }

  TensorDim previous_dim;
  for (unsigned int i = 0; i < layers_name.size(); i++) {
    bool last = false;
    l_type =
      iniparser_getstring(ini, (layers_name[i] + ":Type").c_str(), unknown);
    t = (LayerType)parseType(l_type, TOKEN_LAYER);
    if (i == layers_name.size() - 1) {
      last = true;
    }
    b_zero =
      iniparser_getboolean(ini, (layers_name[i] + ":Bias_zero").c_str(), true);

    switch (t) {
    case LAYER_IN: {
      std::shared_ptr<InputLayer> input_layer = std::make_shared<InputLayer>();

      std::string previous_input_string = iniparser_getstring(
        ini, (layers_name[i] + ":Input_Shape").c_str(), unknown);

      if (previous_input_string == "Unknown") {
        status = ML_ERROR_INVALID_PARAMETER;
        NN_INI_RETURN_STATUS();
      }

      status = previous_dim.setTensorDim(previous_input_string);
      NN_INI_RETURN_STATUS();

      input_layer->setInputDimension(previous_dim);

      status = input_layer->initialize(last);
      NN_INI_RETURN_STATUS();
      input_layer->setBiasZero(b_zero);

      input_layer->setNormalization(iniparser_getboolean(
        ini, (layers_name[i] + ":Normalization").c_str(), false));
      input_layer->setStandardization(iniparser_getboolean(
        ini, (layers_name[i] + ":Standardization").c_str(), false));
      layers.push_back(input_layer);
    } break;
    case LAYER_CONV2D: {
      int size[CONV2D_DIM];
      WeightDecayParam weight_decay;
      std::shared_ptr<Conv2DLayer> conv2d_layer =
        std::make_shared<Conv2DLayer>();

      std::string previous_input_string = iniparser_getstring(
        ini, (layers_name[i] + ":Input_Shape").c_str(), unknown);

      if (previous_input_string.compare("Unknown") != 0) {
        TensorDim d;
        d.setTensorDim(previous_input_string);
        previous_dim = d;
      }

      conv2d_layer->setInputDimension(previous_dim);
      NN_INI_RETURN_STATUS();

      if (last) {
        status = conv2d_layer->setCost(cost);
        NN_INI_RETURN_STATUS();
      }

      conv2d_layer->setBiasZero(b_zero);
      status = conv2d_layer->setActivation((ActiType)parseType(
        iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(),
                            unknown),
        TOKEN_ACTI));
      NN_INI_RETURN_STATUS();
      conv2d_layer->setWeightInit((WeightIniType)parseType(
        iniparser_getstring(ini, (layers_name[i] + ":WeightIni").c_str(),
                            unknown),
        TOKEN_WEIGHTINI));

      status = setWeightDecay(ini, layers_name[i], weight_decay);
      NN_INI_RETURN_STATUS();

      conv2d_layer->setWeightDecay(weight_decay);

      status =
        getValues(CONV2D_DIM,
                  iniparser_getstring(
                    ini, (layers_name[i] + ":kernel_size").c_str(), unknown),
                  (int *)size);
      NN_INI_RETURN_STATUS();
      status = conv2d_layer->setSize(
        size, Layer::PropertyType::kernel_size);
      NN_INI_RETURN_STATUS();

      status = getValues(
        CONV2D_DIM,
        iniparser_getstring(ini, (layers_name[i] + ":stride").c_str(), unknown),
        (int *)size);
      NN_INI_RETURN_STATUS();
      status = conv2d_layer->setSize(
        size, Layer::PropertyType::stride);
      NN_INI_RETURN_STATUS();

      status = getValues(CONV2D_DIM,
                         iniparser_getstring(
                           ini, (layers_name[i] + ":padding").c_str(), unknown),
                         (int *)size);
      NN_INI_RETURN_STATUS();
      status = conv2d_layer->setSize(
        size, Layer::PropertyType::padding);
      NN_INI_RETURN_STATUS();

      status = conv2d_layer->setFilter(
        iniparser_getint(ini, (layers_name[i] + ":filter").c_str(), 0));
      NN_INI_RETURN_STATUS();

      status = conv2d_layer->initialize(last);
      NN_INI_RETURN_STATUS();
      layers.push_back(conv2d_layer);
    } break;
    case LAYER_FC: {
      WeightDecayParam weight_decay;
      std::shared_ptr<FullyConnectedLayer> fc_layer =
        std::make_shared<FullyConnectedLayer>();

      fc_layer->setInputDimension(previous_dim);

      fc_layer->setUnit(static_cast<unsigned int>(
        iniparser_getint(ini, (layers_name[i] + ":Unit").c_str(), 0)));

      if (last) {
        status = fc_layer->setCost(cost);
        NN_INI_RETURN_STATUS();
      }

      if (i == 0) {
        ml_loge("Error: Fully Connected Layer should be after "
                "InputLayer.");
        return ML_ERROR_INVALID_PARAMETER;
      }

      fc_layer->setWeightInit((WeightIniType)parseType(
        iniparser_getstring(ini, (layers_name[i] + ":WeightIni").c_str(),
                            unknown),
        TOKEN_WEIGHTINI));

      status = fc_layer->initialize(last);
      NN_INI_RETURN_STATUS();
      fc_layer->setBiasZero(b_zero);

      status = fc_layer->setOptimizer(opt);
      NN_INI_RETURN_STATUS();

      status = fc_layer->setActivation((ActiType)parseType(
        iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(),
                            unknown),
        TOKEN_ACTI));
      NN_INI_RETURN_STATUS();

      status = setWeightDecay(ini, layers_name[i], weight_decay);
      NN_INI_RETURN_STATUS();

      fc_layer->setWeightDecay(weight_decay);
      layers.push_back(fc_layer);
    } break;
    case LAYER_BN: {
      WeightDecayParam weight_decay;
      std::shared_ptr<BatchNormalizationLayer> bn_layer =
        std::make_shared<BatchNormalizationLayer>();

      bn_layer->setInputDimension(previous_dim);

      status = bn_layer->initialize(last);
      NN_INI_RETURN_STATUS();
      bn_layer->setBiasZero(b_zero);

      status = bn_layer->setOptimizer(opt);
      NN_INI_RETURN_STATUS();
      layers.push_back(bn_layer);
      if (i == 0) {
        ml_loge("Error: BN layer shouldn't be first layer of network");
        return ML_ERROR_INVALID_PARAMETER;
      }
      layers[i - 1]->setBNfollow(true);
      status = bn_layer->setActivation((ActiType)parseType(
        iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(),
                            unknown),
        TOKEN_ACTI));
      NN_INI_RETURN_STATUS();
      status = setWeightDecay(ini, layers_name[i], weight_decay);
      NN_INI_RETURN_STATUS();
      bn_layer->setWeightDecay(weight_decay);
    } break;
    case LAYER_UNKNOWN:
      ml_loge("Error: Unknown layer type");
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    default:
      break;
    }
    previous_dim = layers[i]->getOutputDimension();
  }

  /** Add the last layer as loss layer */
  status = initLossLayer();
  NN_INI_RETURN_STATUS();

  status = data_buffer->setMiniBatch(batch_size);
  NN_INI_RETURN_STATUS();

  status = data_buffer->setBufSize(
    iniparser_getint(ini, "DataSet:BufferSize", batch_size));

  initialized = true;
  iniparser_freedict(ini);
  return status;
}

int NeuralNetwork::initLossLayer() {
  int status = ML_ERROR_NONE;
  CostType updated_cost = cost;
  ActiType act = layers.back()->getActivationType();

  std::shared_ptr<LossLayer> loss_layer = std::make_shared<LossLayer>();
  loss_layer->setInputDimension(layers.back()->getOutputDimension());
  status = loss_layer->initialize(true);
  NN_RETURN_STATUS();

  if (updated_cost == COST_ENTROPY) {
    switch (act) {
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

    act = ACT_NONE;
    status = layers.back()->setActivation(act);
    NN_RETURN_STATUS();
    status = layers.back()->setCost(updated_cost);
    NN_RETURN_STATUS();
  }

  status = loss_layer->setCost(updated_cost);
  NN_RETURN_STATUS();
  status = loss_layer->setActivation(act);
  NN_RETURN_STATUS();

  layers.push_back(loss_layer);
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
    case PropertyType::cost:
    case PropertyType::loss: {
      cost = (CostType)parseType(value, TOKEN_COST);
    } break;
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
    case PropertyType::epochs: {
      int e;
      status = setInt(e, value);
      NN_RETURN_STATUS();
      epoch = e;
    } break;
    case PropertyType::train_data: {
      status = std::static_pointer_cast<DataBufferFromDataFile>(data_buffer)
                 ->setDataFile(value, DATA_TRAIN);
      NN_RETURN_STATUS();
    } break;
    case PropertyType::val_data: {
      status = std::static_pointer_cast<DataBufferFromDataFile>(data_buffer)
                 ->setDataFile(value, DATA_VAL);
      NN_RETURN_STATUS();
    } break;
    case PropertyType::test_data: {
      status = std::static_pointer_cast<DataBufferFromDataFile>(data_buffer)
                 ->setDataFile(value, DATA_TEST);
      NN_RETURN_STATUS();
    } break;
    case PropertyType::label_data: {
      status = std::static_pointer_cast<DataBufferFromDataFile>(data_buffer)
                 ->setDataFile(value, DATA_LABEL);
      NN_RETURN_STATUS();
    } break;

    case PropertyType::buffer_size: {
      int size;
      status = setInt(size, value);
      NN_RETURN_STATUS();
      status = data_buffer->setBufSize(size);
      NN_RETURN_STATUS();
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
      break;
    }
  }

  return status;
}

int NeuralNetwork::init(std::shared_ptr<Optimizer> optimizer,
                        std::vector<std::string> arg_list) {
  int status = ML_ERROR_NONE;
  bool last = false;
  TensorDim previous_dim;
  opt = *optimizer.get();
  status = setProperty(arg_list);
  NN_RETURN_STATUS();

  loss = 10000.0;
  for (unsigned int i = 0; i < layers.size(); ++i) {
    if (i == layers.size() - 1)
      last = true;
    switch (layers[i]->getType()) {
    case LAYER_IN:
      layers[0]->initialize(last);
      previous_dim = layers[0]->getOutputDimension();
      NN_RETURN_STATUS();
      break;
    case LAYER_CONV2D:
      if (i != 0) {
        layers[i]->setInputDimension(previous_dim);
      }
      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();

      status = layers[i]->setCost(cost);
      NN_RETURN_STATUS();
      break;
    case LAYER_FC: {
      std::shared_ptr<FullyConnectedLayer> fc_layer =
        std::static_pointer_cast<FullyConnectedLayer>(layers[i]);
      layers[i]->setInputDimension(previous_dim);

      status = layers[i]->setCost(cost);
      NN_RETURN_STATUS();

      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();

      status = fc_layer->setOptimizer(opt);
      NN_RETURN_STATUS();

    }  break;
    case LAYER_BN:
      layers[i]->setInputDimension(previous_dim);
      status = layers[i]->initialize(last);
      NN_RETURN_STATUS();
      status = layers[i]->setOptimizer(opt);
      NN_RETURN_STATUS();
      layers[i - 1]->setBNfollow(true);
      break;
    default:
      break;
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
  for (unsigned int i = 0; i < layers.size(); i++) {
    layers.erase(layers.begin() + i);
  }

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

  X = forwarding (input, status);
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
  loss = 0.0;
  for (unsigned int i=0; i < layers.size(); i++) {
    loss += layers[i]->getLoss();
  }

  return loss;
}

void NeuralNetwork::setLoss(float l) { loss = l; }

NeuralNetwork &NeuralNetwork::copy(NeuralNetwork &from) {
  if (this != &from) {
    batch_size = from.batch_size;
    learning_rate = from.learning_rate;
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
    data_buffer = std::make_shared<DataBufferFromDataFile>();
  }

  if (values.size() != 0) {
    status = data_buffer->setMiniBatch(layers[0]->getInputDimension().batch());
    NN_RETURN_STATUS();

    status = setProperty(values);
    NN_RETURN_STATUS();
  }

  status = data_buffer->setClassNum(
    layers[layers.size() - 1]->getOutputDimension().width());
  NN_RETURN_STATUS();

  status = data_buffer->setFeatureSize(layers[0]->getInputDimension());
  NN_RETURN_STATUS();

  status = data_buffer->init();
  NN_RETURN_STATUS();

  return train_run();
}

/**
 * @brief     Run NeuralNetwork train
 */
int NeuralNetwork::train(
  std::function<bool(float *, float *, int *)> train_func,
  std::function<bool(float *, float *, int *)> val_func,
  std::function<bool(float *, float *, int *)> test_func) {

  std::vector<std::string> values;

  return train(train_func, val_func, test_func, values);
}

/**
 * @brief     Run NeuralNetwork train
 */
int NeuralNetwork::train(
  std::function<bool(float *, float *, int *)> train_func,
  std::function<bool(float *, float *, int *)> val_func,
  std::function<bool(float *, float *, int *)> test_func,
  std::vector<std::string> values) {

  int status = ML_ERROR_NONE;

  if (data_buffer == nullptr) {
    data_buffer = std::make_shared<DataBufferFromCallback>();

    status = data_buffer->setMiniBatch(layers[0]->getInputDimension().batch());
    NN_RETURN_STATUS();

    status = setProperty(values);
    NN_RETURN_STATUS();
  }

  status = data_buffer->setClassNum(
    layers[layers.size() - 1]->getOutputDimension().width());
  NN_RETURN_STATUS();

  status = data_buffer->setFeatureSize(layers[0]->getInputDimension());
  NN_RETURN_STATUS();

  status = data_buffer->init();
  NN_RETURN_STATUS();

  std::shared_ptr<DataBufferFromCallback> callback_buffer =
    std::static_pointer_cast<DataBufferFromCallback>(data_buffer);

  status = callback_buffer->setFunc(nntrainer::BUF_TRAIN, (train_func));
  if (status != ML_ERROR_NONE)
    return status;

  status = callback_buffer->setFunc(nntrainer::BUF_VAL, (val_func));
  if (status != ML_ERROR_NONE)
    return status;

  status = callback_buffer->setFunc(nntrainer::BUF_TEST, (test_func));
  if (status != ML_ERROR_NONE)
    return status;

  return train_run();
};

/**
 * @brief     Run NeuralNetwork train with callback function by user
 */
int NeuralNetwork::train_run() {
  int status = ML_ERROR_NONE;

  float training_loss = 0.0;
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

    while (true) {
      vec_4d in, label;
      if (data_buffer->getDataFromBuffer(nntrainer::BUF_TRAIN, in, label)) {
        status =
          backwarding(nntrainer::Tensor(in), nntrainer::Tensor(label), iter++);
        if (status != ML_ERROR_NONE) {
          data_buffer->clear(nntrainer::BUF_TRAIN);
          ml_loge ("Error: training error in #%d/%d.", i+1, epoch);
          return status;
        }
        std::cout << "#" << i + 1 << "/" << epoch;
        data_buffer->displayProgress(iter, nntrainer::BUF_TRAIN, getLoss());
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
      float valloss = 0.0;
      int tcases = 0;

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
              ml_loge ("Error: forwarding the network resulted in error.");
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

      valloss = valloss / (float)(tcases);
      std::cout << " >> [ Accuracy: " << right / (float)(tcases)*100.0
                << "% - Validation Loss : " << valloss << " ] ";
    }
    std::cout << std::endl;
  }

  return status;
}

int NeuralNetwork::checkValidation() {
  int status = ML_ERROR_NONE;
  if (!config.empty())
    return status;
  if (layers.size()) {
    return ML_ERROR_INVALID_PARAMETER;
  } else {
    for (std::vector<std::shared_ptr<nntrainer::Layer>>::iterator layer =
           layers.begin();
         layer != layers.end(); ++layer) {
      if (!(*layer)->checkValidation())
        return ML_ERROR_INVALID_PARAMETER;
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
  } else {
    layers.push_back(layer);
  }

  return status;
}

} /* namespace nntrainer */
