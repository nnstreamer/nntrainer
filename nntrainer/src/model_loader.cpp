// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	model_loader.c
 * @date	5 August 2020
 * @brief	This is model loader class for the Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <databuffer_file.h>
#include <databuffer_func.h>
#include <model_loader.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <sstream>
#include <util_func.h>

#define NN_INI_RETURN_STATUS()     \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      iniparser_freedict(ini);     \
      return status;               \
    }                              \
  } while (0)

namespace nntrainer {

/**
 * @brief     load model config from ini
 */
int ModelLoader::loadModelConfigIni(dictionary *ini, NeuralNetwork &model) {
  int status = ML_ERROR_NONE;

  /** Default to neural network model type */
  model.net_type = (nntrainer::NetType)parseType(
    iniparser_getstring(ini, "Model:Type", unknown), TOKEN_MODEL);
  model.epochs = iniparser_getint(ini, "Model:Epochs", model.epochs);
  model.cost = (CostType)parseType(
    iniparser_getstring(ini, "Model:Cost", unknown), TOKEN_COST);
  model.save_path = iniparser_getstring(ini, "Model:Save_path", "./model.bin");
  model.batch_size =
    iniparser_getint(ini, "Model:Batch_Size", model.batch_size);

  /** Default to adam optimizer */
  status = model.opt.setType((OptType)parseType(
    iniparser_getstring(ini, "Model:Optimizer", "adam"), TOKEN_OPT));
  NN_RETURN_STATUS();

  OptParam popt(model.opt.getType());
  popt.learning_rate =
    iniparser_getdouble(ini, "Model:Learning_rate", popt.learning_rate);
  popt.decay_steps =
    iniparser_getint(ini, "Model:Decay_steps", popt.decay_steps);
  popt.decay_rate =
    iniparser_getdouble(ini, "Model:Decay_rate", popt.decay_rate);
  popt.beta1 = iniparser_getdouble(ini, "Model:beta1", popt.beta1);
  popt.beta2 = iniparser_getdouble(ini, "Model:beta2", popt.beta2);
  popt.epsilon = iniparser_getdouble(ini, "Model:epsilon", popt.epsilon);

  status = model.opt.setOptParam(popt);
  NN_RETURN_STATUS();

  return status;
}

/**
 * @brief     load dataset config from ini
 */
int ModelLoader::loadDatasetConfigIni(dictionary *ini, NeuralNetwork &model) {
  /// @fixme: 370
  ml_logd("start parsing dataset config");
  int status = ML_ERROR_NONE;

  if (iniparser_find_entry(ini, "DataSet:Tflite")) {
    ml_loge("Error: Tflite dataset is not yet implemented!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  model.data_buffer = std::make_shared<DataBufferFromDataFile>();
  std::shared_ptr<DataBufferFromDataFile> dbuffer =
    std::static_pointer_cast<DataBufferFromDataFile>(model.data_buffer);

  std::function<int(const char *, DataType, bool)> parse_and_set =
    [&](const char *key, DataType dt, bool required) -> int {
    const char *path = iniparser_getstring(ini, key, NULL);

    if (path == NULL) {
      return required ? ML_ERROR_INVALID_PARAMETER : ML_ERROR_NONE;
    }

    return dbuffer->setDataFile(path, dt);
  };

  status = parse_and_set("DataSet:TrainData", DATA_TRAIN, true);
  NN_RETURN_STATUS();
  status = parse_and_set("DataSet:ValidData", DATA_VAL, false);
  NN_RETURN_STATUS();
  status = parse_and_set("DataSet:TestData", DATA_TEST, false);
  NN_RETURN_STATUS();
  status = parse_and_set("Dataset:LabelData", DATA_LABEL, true);
  NN_RETURN_STATUS();

  /// fixme: #299, #389
  int bufsize = iniparser_getint(ini, "DataSet:BufferSize", model.batch_size);
  ml_logd("buf size: %d", bufsize);
  status = model.data_buffer->setBufSize(bufsize);
  NN_RETURN_STATUS();

  ml_logd("parsing dataset done");
  return status;
}

int ModelLoader::loadLayerConfigIni(dictionary *ini,
                                    std::shared_ptr<Layer> &layer,
                                    std::string layer_name) {
  int status = ML_ERROR_NONE;

  std::string layer_type_str =
    iniparser_getstring(ini, (layer_name + ":Type").c_str(), unknown);
  LayerType layer_type = (LayerType)parseType(layer_type_str, TOKEN_LAYER);

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
    ml_loge("Error: Unknown layer type from %s, parsed to %d",
            layer_type_str.c_str(), layer_type);
    status = ML_ERROR_INVALID_PARAMETER;
    NN_RETURN_STATUS();
  }

  unsigned int property_end =
    static_cast<unsigned int>(Layer::PropertyType::unknown);

  for (unsigned int i = 0; i < property_end; ++i) {
    std::string prop = propToStr(i);
    std::string value =
      iniparser_getstring(ini, (layer_name + ":" + prop).c_str(), unknown);

    /**! @todo: add following negative tc after #319
     * 1. layer has empty prop -> throw std::invalid_argument
     * 2. layer has not allowed property -> throw exception::not_supported
     * 3. property value parse error -> throw std::invalid_argument
     */
    if (!strncmp(value.c_str(), unknown, strlen(unknown))) {
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
  NN_RETURN_STATUS();

  return ML_ERROR_NONE;
}

/**
 * @brief     load all of model and dataset from ini
 */
int ModelLoader::loadFromIni(std::string ini_file, NeuralNetwork &model) {
  int status = ML_ERROR_NONE;
  int num_ini_sec = 0;
  dictionary *ini;
  const char model_str[] = "model";
  unsigned int model_len = strlen(model_str);
  const char dataset_str[] = "dataset";
  unsigned int dataset_len = strlen(dataset_str);

  if (ini_file.empty()) {
    ml_loge("Error: Configuration File is not defined");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (!isFileExist(ini_file)) {
    ml_loge("Cannot open model configuration file, filename : %s",
            ini_file.c_str());
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
    status = ML_ERROR_INVALID_PARAMETER;
    NN_INI_RETURN_STATUS();
  }

  if (iniparser_find_entry(ini, model_str) == 0) {
    ml_loge("there is no [Model] section in given ini file");
    status = ML_ERROR_INVALID_PARAMETER;
    NN_INI_RETURN_STATUS();
  }

  ml_logd("parsing ini started");
  /** Get all the section names */
  ml_logi("==========================parsing ini...");
  ml_logi("invalid properties does not cause error, rather be ignored");
  ml_logi("not-allowed property for the layer throws error");
  ml_logi("valid property with invalid value throws error as well");
  for (int idx = 0; idx < num_ini_sec; ++idx) {
    const char *sec_name = iniparser_getsecname(ini, idx);
    ml_logd("probing section name: %s", sec_name);

    if (!sec_name) {
      ml_loge("Error: Unable to retrieve section names from ini.");
      status = ML_ERROR_INVALID_PARAMETER;
      NN_INI_RETURN_STATUS();
    }

    if (strncasecmp(model_str, sec_name, model_len) == 0) {
      status = loadModelConfigIni(ini, model);
      NN_INI_RETURN_STATUS();
      continue;
    }

    if (strncasecmp(dataset_str, sec_name, dataset_len) == 0) {
      status = loadDatasetConfigIni(ini, model);
      NN_INI_RETURN_STATUS();
      continue;
    }

    /** Parse all the layers defined as sections in order */
    std::shared_ptr<Layer> layer;
    status = loadLayerConfigIni(ini, layer, sec_name);
    NN_INI_RETURN_STATUS();

    status = model.addLayer(layer);
    NN_INI_RETURN_STATUS();
  }
  ml_logd("parsing ini finished");

  /**< Additional validation and handling for the model */
  if (!model.data_buffer) {
    model.data_buffer = std::make_shared<DataBufferFromCallback>();
  }

  status = model.data_buffer->setBatchSize(model.batch_size);
  NN_INI_RETURN_STATUS();

  if (model.layers.empty()) {
    ml_loge("there is no layer section in the ini file");
    status = ML_ERROR_INVALID_PARAMETER;
  }

  iniparser_freedict(ini);
  return status;
}

/**
 * @brief     load all of model and dataset from given config file
 */
int ModelLoader::loadFromConfig(std::string config, NeuralNetwork &model) {
  size_t position = config.find_last_of(".");
  if (position == std::string::npos)
    throw std::invalid_argument("Extension missing in config file");

  if (config.substr(position + 1) == "ini") {
    return loadFromIni(config, model);
  }

  return ML_ERROR_INVALID_PARAMETER;
}

} // namespace nntrainer
