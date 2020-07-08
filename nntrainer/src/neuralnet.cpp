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

static int parseWeightDecay(dictionary *ini, std::string layer_name,
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

NeuralNetwork::NeuralNetwork(std::string config) :
  batch_size(1),
  epoch(1),
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

int NeuralNetwork::loadFromConfig() {
  int status = ML_ERROR_NONE;
  std::string ini_file = config;
  int num_ini_sec = 0;
  char unknown[] = "Unknown";
  char model_name[] = "model.bin";
  dictionary *ini;
  std::vector<std::string> section_names;
  std::vector<std::string>::iterator section_names_iter;

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

  /** Get all the section names */
  for (int idx = 0; idx < num_ini_sec; ++idx) {
    const char *sec_name = iniparser_getsecname(ini, idx);
    if (!sec_name) {
      ml_loge("Error: Unable to retrieve section names from ini.");
      return ML_ERROR_INVALID_PARAMETER;
    }
    std::string sec_name_lower(sec_name);
    std::transform(sec_name_lower.begin(), sec_name_lower.end(),
                   sec_name_lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    section_names.push_back(sec_name_lower);
  }

  /** Parse the Network section and its properties */
  section_names_iter =
    std::find(section_names.begin(), section_names.end(), "network");
  if (section_names_iter == section_names.end()) {
    ml_loge("Error: Network section not found in the .");
    return ML_ERROR_INVALID_PARAMETER;
  } else {
    section_names.erase(section_names_iter);
  }

  /** Default to neural network model type */
  net_type = (nntrainer::NetType)parseType(
    iniparser_getstring(ini, "Network:Type", unknown), TOKEN_NET);
  epoch = iniparser_getint(ini, "Network:Epoch", epoch);
  cost = (CostType)parseType(iniparser_getstring(ini, "Network:Cost", unknown),
                             TOKEN_COST);
  model = iniparser_getstring(ini, "Network:Model", model_name);
  batch_size = iniparser_getint(ini, "Network:Minibatch", batch_size);

  /** Default to adam optimizer */
  status = opt.setType((OptType)parseType(
    iniparser_getstring(ini, "Network:Optimizer", unknown), TOKEN_OPT));
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

  /** Parse the DataSet section */
  section_names_iter =
    std::find(section_names.begin(), section_names.end(), "dataset");
  if (section_names_iter != section_names.end()) {
    section_names.erase(section_names_iter);

    if (iniparser_find_entry(ini, "DataSet:Tflite")) {
      ml_loge("Error: Tflite dataset is not yet implemented!");
      return ML_ERROR_INVALID_PARAMETER;
    } else {
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
      status = data_buffer->setBufSize(
        iniparser_getint(ini, "DataSet:BufferSize", batch_size));
      NN_INI_RETURN_STATUS();
    }
  } else {
    data_buffer = std::make_shared<DataBufferFromCallback>();
  }

  /** Parse all the layers defined as sections in order */
  for (section_names_iter = section_names.begin();
       section_names_iter != section_names.end(); ++section_names_iter) {
    std::string layer_name = *section_names_iter;
    std::string layer_type_str =
      iniparser_getstring(ini, (layer_name + ":Type").c_str(), unknown);
    LayerType layer_type = (LayerType)parseType(layer_type_str, TOKEN_LAYER);
    bool b_zero =
      iniparser_getboolean(ini, (layer_name + ":bias_init_zero").c_str(), true);

    switch (layer_type) {
    case LAYER_IN: {
      std::shared_ptr<InputLayer> input_layer = std::make_shared<InputLayer>();

      std::string input_shape_str = iniparser_getstring(
        ini, (layer_name + ":Input_Shape").c_str(), unknown);

      if (input_shape_str.compare("Unknown") == 0) {
        status = ML_ERROR_INVALID_PARAMETER;
        NN_INI_RETURN_STATUS();
      }

      TensorDim d;
      status = d.setTensorDim(input_shape_str);
      NN_INI_RETURN_STATUS();
      input_layer->setInputDimension(d);

      input_layer->setNormalization(iniparser_getboolean(
        ini, (layer_name + ":Normalization").c_str(), false));
      input_layer->setStandardization(iniparser_getboolean(
        ini, (layer_name + ":Standardization").c_str(), false));
      addLayer(input_layer);
    } break;
    case LAYER_CONV2D: {
      int size[CONV2D_DIM];
      WeightDecayParam weight_decay;
      std::shared_ptr<Conv2DLayer> conv2d_layer =
        std::make_shared<Conv2DLayer>();

      std::string input_shape_str = iniparser_getstring(
        ini, (layer_name + ":Input_Shape").c_str(), unknown);

      if (input_shape_str.compare("Unknown") != 0) {
        TensorDim d;
        d.setTensorDim(input_shape_str);
        conv2d_layer->setInputDimension(d);
      } else if (section_names_iter == section_names.begin()) {
        ml_loge("Error: %s layer input shape not specified.",
                layer_name.c_str());
        status = ML_ERROR_INVALID_PARAMETER;
        NN_INI_RETURN_STATUS();
      }

      status = getValues(CONV2D_DIM,
                         iniparser_getstring(
                           ini, (layer_name + ":kernel_size").c_str(), unknown),
                         (int *)size);
      NN_INI_RETURN_STATUS();
      status = conv2d_layer->setSize(size, Layer::PropertyType::kernel_size);
      NN_INI_RETURN_STATUS();

      status =
        getValues(CONV2D_DIM,
                  iniparser_getstring(ini, (layer_name + ":stride").c_str(),
                                      getValues({1, 1})),
                  (int *)size);
      NN_INI_RETURN_STATUS();
      status = conv2d_layer->setSize(size, Layer::PropertyType::stride);
      NN_INI_RETURN_STATUS();

      status =
        getValues(CONV2D_DIM,
                  iniparser_getstring(ini, (layer_name + ":padding").c_str(),
                                      getValues({0, 0})),
                  (int *)size);

      NN_INI_RETURN_STATUS();
      status = conv2d_layer->setSize(size, Layer::PropertyType::padding);
      NN_INI_RETURN_STATUS();

      status = conv2d_layer->setFilter(
        iniparser_getint(ini, (layer_name + ":filter").c_str(), 0));
      NN_INI_RETURN_STATUS();

      conv2d_layer->setBiasZero(b_zero);
      conv2d_layer->setWeightInit((WeightIniType)parseType(
        iniparser_getstring(ini, (layer_name + ":WeightIni").c_str(),
                            "xavier_uniform"),
        TOKEN_WEIGHTINI));

      status = parseWeightDecay(ini, layer_name, weight_decay);
      NN_INI_RETURN_STATUS();

      conv2d_layer->setWeightDecay(weight_decay);
      NN_INI_RETURN_STATUS();

      addLayer(conv2d_layer);
    } break;

    case LAYER_POOLING2D: {
      int size[POOLING2D_DIM];
      std::shared_ptr<Pooling2DLayer> pooling2d_layer =
        std::make_shared<Pooling2DLayer>();

      status = getValues(
        POOLING2D_DIM,
        iniparser_getstring(ini, (layer_name + ":pooling_size").c_str(),
                            getValues({1, 1})),
        (int *)size);

      NN_INI_RETURN_STATUS();
      status =
        pooling2d_layer->setSize(size, Layer::PropertyType::pooling_size);

      NN_INI_RETURN_STATUS();
      status =
        getValues(POOLING2D_DIM,
                  iniparser_getstring(ini, (layer_name + ":stride").c_str(),
                                      getValues({1, 1})),
                  (int *)size);
      NN_INI_RETURN_STATUS();
      status = pooling2d_layer->setSize(size, Layer::PropertyType::stride);
      NN_INI_RETURN_STATUS();
      status =
        getValues(POOLING2D_DIM,
                  iniparser_getstring(ini, (layer_name + ":padding").c_str(),
                                      getValues({0, 0})),
                  (int *)size);
      NN_INI_RETURN_STATUS();
      status = pooling2d_layer->setSize(size, Layer::PropertyType::padding);
      NN_INI_RETURN_STATUS();

      pooling2d_layer->setPoolingType(
        (nntrainer::Pooling2DLayer::PoolingType)parseType(
          iniparser_getstring(ini, (layer_name + ":pooling").c_str(),
                              "average"),
          TOKEN_POOLING));

      addLayer(pooling2d_layer);
    } break;

    case LAYER_FLATTEN: {
      std::shared_ptr<FlattenLayer> flatten_layer =
        std::make_shared<FlattenLayer>();

      addLayer(flatten_layer);
    } break;

    case LAYER_FC: {
      WeightDecayParam weight_decay;
      std::shared_ptr<FullyConnectedLayer> fc_layer =
        std::make_shared<FullyConnectedLayer>();

      std::string input_shape_str = iniparser_getstring(
        ini, (layer_name + ":Input_Shape").c_str(), unknown);

      if (input_shape_str.compare("Unknown") != 0) {
        TensorDim d;
        d.setTensorDim(input_shape_str);
        fc_layer->setInputDimension(d);
      } else if (section_names_iter == section_names.begin()) {
        ml_loge("Error: %s layer input shape not specified.",
                layer_name.c_str());
        status = ML_ERROR_INVALID_PARAMETER;
        NN_INI_RETURN_STATUS();
      }

      fc_layer->setUnit(static_cast<unsigned int>(
        iniparser_getint(ini, (layer_name + ":Unit").c_str(), 0)));

      fc_layer->setBiasZero(b_zero);
      fc_layer->setWeightInit((WeightIniType)parseType(
        iniparser_getstring(ini, (layer_name + ":WeightIni").c_str(),
                            "xavier_uniform"),
        TOKEN_WEIGHTINI));

      status = parseWeightDecay(ini, layer_name, weight_decay);
      NN_INI_RETURN_STATUS();

      fc_layer->setWeightDecay(weight_decay);

      addLayer(fc_layer);
    } break;
    case LAYER_BN: {
      std::shared_ptr<BatchNormalizationLayer> bn_layer =
        std::make_shared<BatchNormalizationLayer>();

      // fixme: deprecate this.
      layers.back()->setBNfollow(true);

      addLayer(bn_layer);
      NN_INI_RETURN_STATUS();
    } break;
    case LAYER_UNKNOWN:
    default:
      ml_loge("Error: Unknown layer type");
      status = ML_ERROR_INVALID_PARAMETER;
      NN_INI_RETURN_STATUS();
      break;
    }

    /** Add activation layer */
    const char *acti_str =
      iniparser_getstring(ini, (layer_name + ":Activation").c_str(), unknown);
    ActiType act = (ActiType)parseType(acti_str, TOKEN_ACTI);
    layers.back()->setActivation(act);

    /** Add flatten layer */
    bool flatten =
      iniparser_getboolean(ini, (layer_name + ":Flatten").c_str(), false);
    layers.back()->setFlatten(flatten);
  }

  status = data_buffer->setMiniBatch(batch_size);
  NN_INI_RETURN_STATUS();

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
    case LAYER_IN:
      layers[i]->initialize(last);
      break;
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
  loss = 0.0;
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
    data_buffer = std::make_shared<DataBufferFromDataFile>();
  }

  if (values.size() != 0) {
    status = data_buffer->setMiniBatch(layers[0]->getInputDimension().batch());
    NN_RETURN_STATUS();

    status = setTrainConfig(values);
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

    status = setTrainConfig(values);
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

} /* namespace nntrainer */
