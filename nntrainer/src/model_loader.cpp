// SPDX-License-Identifier: Apache-2.0
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

#include <adam.h>
#include <databuffer_factory.h>
#include <databuffer_file.h>
#include <databuffer_func.h>
#include <layer_factory.h>
#include <model_loader.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_factory.h>
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

  if (iniparser_find_entry(ini, "Model") == 0) {
    ml_loge("there is no [Model] section in given ini file");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** Default to neural network model type */
  model.net_type = (nntrainer::NetType)parseType(
    iniparser_getstring(ini, "Model:Type", unknown), TOKEN_MODEL);
  model.epochs = iniparser_getint(ini, "Model:Epochs", model.epochs);
  model.loss_type = (LossType)parseType(
    iniparser_getstring(ini, "Model:Loss", unknown), TOKEN_LOSS);
  model.save_path = iniparser_getstring(ini, "Model:Save_path", "./model.bin");
  model.batch_size =
    iniparser_getint(ini, "Model:Batch_Size", model.batch_size);

  /** Default to adam optimizer */
  OptType opt_type = (OptType)parseType(
    iniparser_getstring(ini, "Model:Optimizer", "adam"), TOKEN_OPT);

  try {
    model.opt = nntrainer::createOptimizer(opt_type);
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("Creating the optimizer failed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::vector<std::string> optimizer_prop = {};
  optimizer_prop.push_back(
    {"learning_rate=" +
     std::string(iniparser_getstring(
       ini, "Model:Learning_rate",
       std::to_string(model.opt->getLearningRate()).c_str()))});

  optimizer_prop.push_back(
    {"decay_steps=" + std::string(iniparser_getstring(
                        ini, "Model:Decay_steps",
                        std::to_string(model.opt->getDecaySteps()).c_str()))});
  optimizer_prop.push_back(
    {"decay_rate=" + std::string(iniparser_getstring(
                       ini, "Model:Decay_rate",
                       std::to_string(model.opt->getDecayRate()).c_str()))});

  if (model.opt->getType() == OptType::ADAM) {
    std::shared_ptr<Adam> opt_adam = std::static_pointer_cast<Adam>(model.opt);

    optimizer_prop.push_back(
      {"beta1=" +
       std::string(iniparser_getstring(
         ini, "Model:Beta1", std::to_string(opt_adam->getBeta1()).c_str()))});
    optimizer_prop.push_back(
      {"beta2=" +
       std::string(iniparser_getstring(
         ini, "Model:Beta2", std::to_string(opt_adam->getBeta2()).c_str()))});
    optimizer_prop.push_back(
      {"epsilon=" + std::string(iniparser_getstring(
                      ini, "Model:Epsilon",
                      std::to_string(opt_adam->getEpsilon()).c_str()))});
  }

  status = model.opt->setProperty(optimizer_prop);
  NN_RETURN_STATUS();

  return status;
}

/**
 * @brief     load dataset config from ini
 */
int ModelLoader::loadDatasetConfigIni(dictionary *ini, NeuralNetwork &model) {
  ml_logd("start parsing dataset config");
  int status = ML_ERROR_NONE;

  if (iniparser_find_entry(ini, "Dataset") == 0) {
    model.data_buffer = nntrainer::createDataBuffer(DataBufferType::GENERATOR);
    status = model.data_buffer->setBatchSize(model.batch_size);
    return status;
  }

  if (iniparser_find_entry(ini, "DataSet:Tflite")) {
    ml_loge("Error: Tflite dataset is not yet implemented!");
    return ML_ERROR_INVALID_PARAMETER;
  }

  model.data_buffer = nntrainer::createDataBuffer(DataBufferType::FILE);
  std::shared_ptr<DataBufferFromDataFile> dbuffer =
    std::static_pointer_cast<DataBufferFromDataFile>(model.data_buffer);

  std::function<int(const char *, DataType, bool)> parse_and_set =
    [&](const char *key, DataType dt, bool required) -> int {
    const char *path = iniparser_getstring(ini, key, NULL);

    if (path == NULL) {
      return required ? ML_ERROR_INVALID_PARAMETER : ML_ERROR_NONE;
    }

    return dbuffer->setDataFile(dt, path);
  };

  status = parse_and_set("DataSet:TrainData", DATA_TRAIN, true);
  NN_RETURN_STATUS();
  status = parse_and_set("DataSet:ValidData", DATA_VAL, false);
  NN_RETURN_STATUS();
  status = parse_and_set("DataSet:TestData", DATA_TEST, false);
  NN_RETURN_STATUS();
  status = parse_and_set("Dataset:LabelData", DATA_LABEL, true);
  NN_RETURN_STATUS();

  status = model.data_buffer->setBatchSize(model.batch_size);
  NN_RETURN_STATUS();

  unsigned int bufsize = iniparser_getint(ini, "DataSet:BufferSize", 1);
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

  try {
    layer = nntrainer::createLayer(layer_type);
  } catch (const std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    status = ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("unknown error type thrown");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  NN_RETURN_STATUS();

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

  status = loadModelConfigIni(ini, model);
  NN_INI_RETURN_STATUS();

  status = loadDatasetConfigIni(ini, model);
  NN_INI_RETURN_STATUS();

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

    if (strncasecmp(model_str, sec_name, model_len) == 0)
      continue;

    if (strncasecmp(dataset_str, sec_name, dataset_len) == 0)
      continue;

    /** Parse all the layers defined as sections in order */
    std::shared_ptr<Layer> layer;
    status = loadLayerConfigIni(ini, layer, sec_name);
    NN_INI_RETURN_STATUS();

    status = model.addLayer(layer);
    NN_INI_RETURN_STATUS();
  }
  ml_logd("parsing ini finished");

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
