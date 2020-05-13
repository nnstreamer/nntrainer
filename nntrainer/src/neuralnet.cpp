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

NeuralNetwork::NeuralNetwork() {
  batch_size = 0;
  learning_rate = 0.0;
  decay_rate = 0.0;
  decay_steps = 0.0;
  epoch = 0;
  loss = 0.0;
  cost = COST_UNKNOWN;
  weight_ini = WEIGHT_UNKNOWN;
  net_type = NET_UNKNOWN;
  data_buffer = NULL;
  config = "";
}

NeuralNetwork::NeuralNetwork(std::string config) {
  batch_size = 0;
  learning_rate = 0.0;
  decay_rate = 0.0;
  decay_steps = 0.0;
  epoch = 0;
  loss = 0.0;
  cost = COST_UNKNOWN;
  weight_ini = WEIGHT_UNKNOWN;
  net_type = NET_UNKNOWN;
  data_buffer = NULL;
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
  weight_ini = (WeightIniType)parseType(
    iniparser_getstring(ini, "Network:WeightIni", unknown), TOKEN_WEIGHTINI);

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

  for (unsigned int i = 0; i < layers_name.size(); ++i) {
    unsigned int h_size;
    l_type =
      iniparser_getstring(ini, (layers_name[i] + ":Type").c_str(), unknown);
    t = (LayerType)parseType(l_type, TOKEN_LAYER);

    switch (t) {
    case LAYER_BN: {
      if (i == 0) {
        ml_loge("Error: Batch NormalizationLayer should be after "
                "InputLayer.");
        return ML_ERROR_INVALID_PARAMETER;
      }
      h_size = hidden_size[i - 1];
    }

    break;
    case LAYER_IN:
    case LAYER_FC:
    default:
      h_size =
        iniparser_getint(ini, (layers_name[i] + ":HiddenSize").c_str(), 0);
      break;
    }
    hidden_size.push_back(h_size);
  }

  if (hidden_size.size() != layers_name.size()) {
    ml_loge("Error: Missing HiddenSize!");
    iniparser_freedict(ini);
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (iniparser_find_entry(ini, "DataSet:TrainData")) {
    data_buffer = new DataBufferFromDataFile();

    DataBufferFromDataFile *dbuffer =
      static_cast<DataBufferFromDataFile *>(data_buffer);
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
    data_buffer = new DataBufferFromCallback();
  }

  status = data_buffer->setMiniBatch(batch_size);
  NN_INI_RETURN_STATUS();

  status = data_buffer->setClassNum(hidden_size[layers_name.size() - 1]);
  NN_INI_RETURN_STATUS();

  status = data_buffer->setBufSize(
    iniparser_getint(ini, "DataSet:BufferSize", batch_size));

  for (unsigned int i = 0; i < layers_name.size(); i++) {
    bool last = false;
    l_type =
      iniparser_getstring(ini, (layers_name[i] + ":Type").c_str(), unknown);
    t = (LayerType)parseType(l_type, TOKEN_LAYER);
    if (i == (layers_name.size() - 1)) {
      last = true;
    }
    b_zero =
      iniparser_getboolean(ini, (layers_name[i] + ":Bias_zero").c_str(), true);
    std::stringstream ss;
    ml_logi("%d : %s %d %d", i, l_type.c_str(), hidden_size[i], b_zero);

    switch (t) {
    case LAYER_IN: {
      std::shared_ptr<InputLayer> input_layer = std::make_shared<InputLayer>();

      input_layer->setType(t);
      status = input_layer->initialize(batch_size, 1, hidden_size[i], last,
                                       b_zero, weight_ini);
      NN_INI_RETURN_STATUS();

      status = input_layer->setOptimizer(opt);
      NN_INI_RETURN_STATUS();

      input_layer->setNormalization(iniparser_getboolean(
        ini, (layers_name[i] + ":Normalization").c_str(), false));
      input_layer->setStandardization(iniparser_getboolean(
        ini, (layers_name[i] + ":Standardization").c_str(), false));
      status = input_layer->setActivation((ActiType)parseType(
        iniparser_getstring(ini, (layers_name[i] + ":Activation").c_str(),
                            unknown),
        TOKEN_ACTI));
      NN_INI_RETURN_STATUS();
      layers.push_back(input_layer);
    } break;
    case LAYER_FC: {
      WeightDecayParam weight_decay;
      std::shared_ptr<FullyConnectedLayer> fc_layer =
        std::make_shared<FullyConnectedLayer>();
      fc_layer->setType(t);
      status = fc_layer->setCost(cost);
      NN_INI_RETURN_STATUS();
      if (i == 0) {
        ml_loge("Error: Fully Connected Layer should be after "
                "InputLayer.");
        return ML_ERROR_INVALID_PARAMETER;
      }
      status = fc_layer->initialize(batch_size, hidden_size[i - 1],
                                    hidden_size[i], last, b_zero, weight_ini);
      NN_INI_RETURN_STATUS();
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
      bn_layer->setType(t);
      status = bn_layer->setOptimizer(opt);
      NN_INI_RETURN_STATUS();
      status = bn_layer->initialize(batch_size, 1, hidden_size[i], last, b_zero,
                                    weight_ini);
      NN_INI_RETURN_STATUS();
      layers.push_back(bn_layer);
      if (i == 0) {
        ml_loge("Error: BN layer shouldn't be first layer of network");
        return ML_ERROR_INVALID_PARAMETER;
      }
      layers[i - 1]->setBNfallow(true);
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
  }

  status = data_buffer->setFeatureSize(hidden_size[0]);
  NN_INI_RETURN_STATUS();

  status = data_buffer->init();
  NN_INI_RETURN_STATUS();

  iniparser_freedict(ini);
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
    delete data_buffer;
  }
}

/**
 * @brief     forward propagation using layers object which has layer
 */
Tensor NeuralNetwork::forwarding(Tensor input, int &status) {
  Tensor X = input;
  for (unsigned int i = 0; i < layers.size(); i++) {
    X = layers[i]->forwarding(X, status);
  }
  return X;
}

/**
 * @brief     forward propagation using layers object which has layer
 */
Tensor NeuralNetwork::forwarding(Tensor input, Tensor output, int &status) {
  Tensor X = input;
  Tensor Y2 = output;
  for (unsigned int i = 0; i < layers.size(); i++) {
    X = layers[i]->forwarding(X, Y2, status);
  }
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
  Tensor Y = forwarding(X, status);

  for (unsigned int i = layers.size() - 1; i > 0; i--) {
    Y2 = layers[i]->backwarding(Y2, iteration);
  }
  return status;
}

float NeuralNetwork::getLoss() {
  std::shared_ptr<FullyConnectedLayer> out =
    std::static_pointer_cast<FullyConnectedLayer>((layers[layers.size() - 1]));
  return out->getLoss();
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
 */
void NeuralNetwork::saveModel() {
  std::ofstream model_file(model, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->save(model_file);
  model_file.close();
}

/**
 * @brief     read model
 *            read Weight & Bias Data into file by calling save from layer
 */
void NeuralNetwork::readModel() {
  if (!is_file_exist(model))
    return;
  std::ifstream model_file(model, std::ios::in | std::ios::binary);
  for (unsigned int i = 0; i < layers.size(); i++)
    layers[i]->read(model_file);
  model_file.close();
  ml_logi("read modelfile: %s", model.c_str());
}

/**
 * @brief     Run NeuralNetwork train
 */
int NeuralNetwork::train(
  std::function<bool(vec_3d &, vec_3d &, int &)> train_func,
  std::function<bool(vec_3d &, vec_3d &, int &)> val_func,
  std::function<bool(vec_3d &, vec_3d &, int &)> test_func) {

  int status = ML_ERROR_NONE;

  DataBufferFromCallback *callback_buffer =
    static_cast<DataBufferFromCallback *>(data_buffer);

  status = callback_buffer->setFunc(nntrainer::BUF_TRAIN, (train_func));
  if (status != ML_ERROR_NONE)
    return status;

  status = callback_buffer->setFunc(nntrainer::BUF_VAL, (val_func));
  if (status != ML_ERROR_NONE)
    return status;

  status = callback_buffer->setFunc(nntrainer::BUF_TEST, (test_func));
  if (status != ML_ERROR_NONE)
    return status;

  return train();
};

/**
 * @brief     Run NeuralNetwork train with callback function by user
 */
int NeuralNetwork::train() {
  int status = ML_ERROR_NONE;
  status = data_buffer->run(nntrainer::BUF_TRAIN);
  if (status != ML_ERROR_NONE) {
    data_buffer->clear(BUF_TRAIN);
    return status;
  }

  status = data_buffer->run(nntrainer::BUF_VAL);
  if (status != ML_ERROR_NONE) {
    data_buffer->clear(BUF_VAL);
    return status;
  }

  status = data_buffer->run(nntrainer::BUF_TEST);
  if (status != ML_ERROR_NONE) {
    data_buffer->clear(BUF_TEST);
    return status;
  }

  float training_loss = 0.0;
  for (unsigned int i = 0; i < epoch; ++i) {
    int count = 0;
    while (true) {
      vec_3d in, label;
      if (data_buffer->getDataFromBuffer(nntrainer::BUF_TRAIN, in, label)) {
        backwarding(nntrainer::Tensor(in), nntrainer::Tensor(label), i);
        count++;
        std::cout << "#" << i + 1 << "/" << epoch;
        data_buffer->displayProgress(count, nntrainer::BUF_TRAIN, getLoss());
      } else {
        data_buffer->clear(nntrainer::BUF_TRAIN);
        status = data_buffer->run(nntrainer::BUF_TRAIN);
        if (status != ML_ERROR_NONE) {
          data_buffer->clear(BUF_TRAIN);
          return status;
        }
        break;
      }
    }
    training_loss = getLoss();

    std::cout << "#" << i + 1 << "/" << epoch
              << " - Training Loss: " << training_loss;

    if (data_buffer->getValidation()[1]) {
      int right = 0;
      float valloss = 0.0;
      int tcases = 0;
      while (true) {
        vec_3d in, label;
        if (data_buffer->getDataFromBuffer(nntrainer::BUF_VAL, in, label)) {
          for (int i = 0; i < batch_size; ++i) {
            nntrainer::Tensor X = nntrainer::Tensor({in[i]});
            nntrainer::Tensor Y2 = nntrainer::Tensor({label[i]});
            nntrainer::Tensor Y = forwarding(X, Y2, status);
            if (Y.argmax() == Y2.argmax())
              right++;
            valloss += getLoss();
            tcases++;
          }
        } else {
          data_buffer->clear(nntrainer::BUF_VAL);
          status = data_buffer->run(nntrainer::BUF_VAL);
          if (status != ML_ERROR_NONE) {
            data_buffer->clear(BUF_VAL);
            return status;
          }
          break;
        }
      }

      valloss = valloss / (float)(tcases);
      std::cout << " >> [ Accuracy: " << right / (float)(tcases)*100.0
                << "% - Validation Loss : " << valloss << " ] ";
    }
    std::cout << std::endl;
    saveModel();
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
  layers.push_back(layer);
  return status;
}

} /* namespace nntrainer */
