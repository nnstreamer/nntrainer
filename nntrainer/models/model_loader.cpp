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
#include <sstream>

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
#include <sgd.h>
#include <util_func.h>

#if defined(ENABLE_NNSTREAMER_BACKBONE)
#include <nnstreamer_layer.h>
#endif

#if defined(ENABLE_TFLITE_BACKBONE)
#include <tflite_layer.h>
#endif

#define NN_INI_RETURN_STATUS()     \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      iniparser_freedict(ini);     \
      return status;               \
    }                              \
  } while (0)

namespace nntrainer {

int ModelLoader::loadOptimizerConfigIni(dictionary *ini, NeuralNetwork &model) {
  int status = ML_ERROR_NONE;

  if (iniparser_find_entry(ini, "Optimizer") == 0) {
    if (!model.opt) {
      ml_logw("there is no [Optimizer] section in given ini file."
              "This model can only be used for inference.");
    }
    return ML_ERROR_NONE;
  }

  /** Optimizer already set with deprecated method */
  if (model.opt) {
    ml_loge("Error: optimizers specified twice.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** Default to adam optimizer */
  const char *opt_type = iniparser_getstring(ini, "Optimizer:Type", "adam");
  std::vector<std::string> properties =
    parseProperties(ini, "Optimizer", {"type"});

  try {
    std::shared_ptr<ml::train::Optimizer> optimizer =
      app_context.createObject<ml::train::Optimizer>(opt_type, properties);
    model.setOptimizer(optimizer);
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("Creating the optimizer failed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

/**
 * @brief     load model config from ini
 */
int ModelLoader::loadModelConfigIni(dictionary *ini, NeuralNetwork &model) {
  int status = ML_ERROR_NONE;

  if (iniparser_find_entry(ini, "Model") == 0) {
    ml_loge("there is no [Model] section in given ini file");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::vector<std::string> properties =
    parseProperties(ini, "Model",
                    {"optimizer", "learning_rate", "decay_steps", "decay_rate",
                     "beta1", "beta2", "epsilon", "type", "save_path"});

  status = model.setProperty(properties);
  if (status != ML_ERROR_NONE)
    return status;

  /** handle save_path as a special case for model_file_context */
  const std::string &save_path =
    iniparser_getstring(ini, "Model:Save_path", unknown);
  if (save_path != unknown) {
    model.setSavePath(resolvePath(save_path));
  }

  /**
   ********
   * Note: Below is only to maintain backward compatibility
   ********
   */

  /** If no optimizer specified, exit without error */
  const char *opt_type = iniparser_getstring(ini, "Model:Optimizer", unknown);
  if (opt_type == unknown)
    return status;

  if (model.opt) {
    /** Optimizer already set with a new section */
    ml_loge("Error: optimizers specified twice.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  ml_logw("Warning: using deprecated ini style for optimizers.");
  ml_logw(
    "Warning: create [ Optimizer ] section in ini to specify optimizers.");

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

  if (model.opt->getType() == SGD::type || model.opt->getType() == Adam::type) {
    std::shared_ptr<OptimizerImpl> opt_impl =
      std::static_pointer_cast<OptimizerImpl>(model.opt);

    optimizer_prop.push_back(
      {"decay_steps=" + std::string(iniparser_getstring(
                          ini, "Model:Decay_steps",
                          std::to_string(opt_impl->getDecaySteps()).c_str()))});
    optimizer_prop.push_back(
      {"decay_rate=" + std::string(iniparser_getstring(
                         ini, "Model:Decay_rate",
                         std::to_string(opt_impl->getDecayRate()).c_str()))});

    if (opt_impl->getType() == "adam") {
      std::shared_ptr<Adam> opt_adam = std::static_pointer_cast<Adam>(opt_impl);

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
  }

  status = model.opt->setProperty(optimizer_prop);
  NN_RETURN_STATUS();

  return status;
}

/**
 * @brief     load dataset config from ini
 */
int ModelLoader::loadDatasetConfigIni(dictionary *ini, NeuralNetwork &model) {
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

    return dbuffer->setDataFile(dt, resolvePath(path));
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

std::vector<std::string>
ModelLoader::parseProperties(dictionary *ini, const std::string &section_name,
                             const std::vector<std::string> &filter_props) {
  int num_entries = iniparser_getsecnkeys(ini, section_name.c_str());

  ml_logd("number of entries for %s: %d", section_name.c_str(), num_entries);

  if (num_entries < 1) {
    std::stringstream ss;
    ss << "there are no entries in the section: " << section_name;
    throw std::invalid_argument(ss.str());
  }

  std::unique_ptr<const char *[]> key_refs(new const char *[num_entries]);

  if (iniparser_getseckeys(ini, section_name.c_str(), key_refs.get()) ==
      nullptr) {
    std::stringstream ss;
    ss << "failed to fetch key for section: " << section_name;
    throw std::invalid_argument(ss.str());
  }

  std::vector<std::string> properties;
  properties.reserve(num_entries - 1);

  for (int i = 0; i < num_entries; ++i) {
    /// key is ini section key, which is section_name + ":" + prop_key
    std::string key(key_refs[i]);
    std::string prop_key = key.substr(key.find(":") + 1);

    bool filter_key_found = false;
    for (auto const &filter_key : filter_props)
      if (istrequal(prop_key, filter_key))
        filter_key_found = true;
    if (filter_key_found)
      continue;

    std::string value = iniparser_getstring(ini, key_refs[i], unknown);

    if (value == unknown) {
      std::stringstream ss;
      ss << "parsing property failed key: " << key;
      throw std::invalid_argument(ss.str());
    }

    if (value == "") {
      std::stringstream ss;
      ss << "property key " << key << " has empty value. It is not allowed";
      throw std::invalid_argument(ss.str());
    }
    ml_logd("parsed properties: %s=%s", prop_key.c_str(), value.c_str());

    properties.push_back(prop_key + "=" + value);
  }

  return properties;
}

int ModelLoader::loadLayerConfigIniCommon(dictionary *ini,
                                          std::shared_ptr<Layer> &layer,
                                          const std::string &layer_name,
                                          const std::string &layer_type) {
  int status = ML_ERROR_NONE;
  std::vector<std::string> properties =
    parseProperties(ini, layer_name, {"type", "backbone"});

  try {
    std::shared_ptr<ml::train::Layer> layer_ =
      app_context.createObject<ml::train::Layer>(layer_type, properties);
    layer = std::static_pointer_cast<Layer>(layer_);
  } catch (const std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    status = ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("unknown error type thrown");
    status = ML_ERROR_INVALID_PARAMETER;
  }
  NN_RETURN_STATUS();

  status = layer->setName(layer_name);
  NN_RETURN_STATUS();

  return ML_ERROR_NONE;
}

int ModelLoader::loadLayerConfigIni(dictionary *ini,
                                    std::shared_ptr<Layer> &layer,
                                    const std::string &layer_name) {
  const std::string &layer_type =
    iniparser_getstring(ini, (layer_name + ":Type").c_str(), unknown);

  return loadLayerConfigIniCommon(ini, layer, layer_name, layer_type);
}

int ModelLoader::loadBackboneConfigIni(dictionary *ini,
                                       const std::string &backbone_config,
                                       NeuralNetwork &model,
                                       const std::string &backbone_name) {
  int status = ML_ERROR_NONE;
  NeuralNetwork backbone;

  status = loadFromConfig(backbone_config, backbone, true);
  NN_RETURN_STATUS();

  /** Wait for #361 Load the backbone from its saved file */
  // bool preload =
  //   iniparser_getboolean(ini, (backbone_name + ":Preload").c_str(), true);

  bool trainable =
    iniparser_getboolean(ini, (backbone_name + ":Trainable").c_str(), false);

  double scale_size =
    iniparser_getdouble(ini, (backbone_name + ":ScaleSize").c_str(), 1.0);
  if (scale_size <= 0.0)
    return ML_ERROR_INVALID_PARAMETER;

  std::string input_layer =
    iniparser_getstring(ini, (backbone_name + ":InputLayer").c_str(), "");
  std::string output_layer =
    iniparser_getstring(ini, (backbone_name + ":OutputLayer").c_str(), "");

  auto graph = backbone.getUnsortedLayers(input_layer, output_layer);

  if (graph.empty()) {
    ml_loge("Empty backbone.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  for (auto &layer : graph) {
    layer->setTrainable(trainable);
    layer->resetDimension();
    if (scale_size != 1) {
      layer->scaleSize(scale_size);
    }
    /** TODO #361: this needs update in model file to be of dictionary format */
    // if (preload) {
    //   layer->weight_initializer = WeightInitializer::FILE_INITIALIZER;
    //   layer->bias_initializer = WeightInitializer::FILE_INITIALIZER;
    //   layer->initializer_file = backbone.save_path;
    // }
  }

  // set input dimension for the first layer in the graph

  /** FIXME :the layers is not the actual model_graph. It is just the vector of
   * layers generated by Model Loader. so graph[0] is still valid. Also we need
   * to consider that the first layer of ini might be th first of layers. Need
   * to change by compiling the backbone before using here. */
  std::string input_shape =
    iniparser_getstring(ini, (backbone_name + ":Input_Shape").c_str(), "");
  if (!input_shape.empty()) {
    graph[0]->setProperty(Layer::PropertyType::input_shape, input_shape);
  }

  std::string input_layers =
    iniparser_getstring(ini, (backbone_name + ":Input_Layers").c_str(), "");
  if (!input_layers.empty() && graph.size() != 0) {
    graph[0]->setProperty(Layer::PropertyType::input_layers, input_layers);
  }

  status = model.extendGraph(graph, backbone_name);
  NN_RETURN_STATUS();

  return ML_ERROR_NONE;
}

int ModelLoader::loadBackboneConfigExternal(dictionary *ini,
                                            const std::string &backbone_config,
                                            std::shared_ptr<Layer> &layer,
                                            const std::string &backbone_name) {
  std::string type;

#if defined(ENABLE_NNSTREAMER_BACKBONE)
  type = NNStreamerLayer::type;
#endif

  /** TfLite has higher priority */
#if defined(ENABLE_TFLITE_BACKBONE)
  if (fileTfLite(backbone_config))
    type = TfLiteLayer::type;
#endif

  int status = ML_ERROR_NONE;
  status = loadLayerConfigIniCommon(ini, layer, backbone_name, type);
  NN_RETURN_STATUS();

  layer->setProperty(Layer::PropertyType::modelfile, backbone_config);
  return status;
}

/**
 * @brief     load all of model and dataset from ini
 */
int ModelLoader::loadFromIni(std::string ini_file, NeuralNetwork &model,
                             bool bare_layers) {
  int status = ML_ERROR_NONE;
  int num_ini_sec = 0;
  dictionary *ini;
  const char model_str[] = "model";
  unsigned int model_len = strlen(model_str);
  const char dataset_str[] = "dataset";
  unsigned int dataset_len = strlen(dataset_str);
  const char optimizer_str[] = "optimizer";
  unsigned int optimizer_len = strlen(optimizer_str);

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

  if (!bare_layers) {
    status = loadModelConfigIni(ini, model);
    NN_INI_RETURN_STATUS();

    status = loadDatasetConfigIni(ini, model);
    NN_INI_RETURN_STATUS();

    status = loadOptimizerConfigIni(ini, model);
    NN_INI_RETURN_STATUS();
  }

  ml_logd("parsing ini started");
  /** Get all the section names */
  ml_logi("==========================parsing ini...");
  ml_logi("not-allowed property for the layer throws error");
  ml_logi("valid property with invalid value throws error as well");
  for (int idx = 0; idx < num_ini_sec; ++idx) {
    std::string sec_name = iniparser_getsecname(ini, idx);
    ml_logd("probing section name: %s", sec_name.c_str());

    if (sec_name.empty()) {
      ml_loge("Error: Unable to retrieve section names from ini.");
      status = ML_ERROR_INVALID_PARAMETER;
      NN_INI_RETURN_STATUS();
    }

    if (strncasecmp(model_str, sec_name.c_str(), model_len) == 0)
      continue;

    if (strncasecmp(dataset_str, sec_name.c_str(), dataset_len) == 0)
      continue;

    if (strncasecmp(optimizer_str, sec_name.c_str(), optimizer_len) == 0)
      continue;

    /** Parse all the layers defined as sections in order */
    std::shared_ptr<Layer> layer;

    /**
     * If this section is a backbone, load backbone section from this
     * @note The order of backbones in the ini file defines the order on the
     * backbones in the model graph
     */
    const char *backbone_path =
      iniparser_getstring(ini, (sec_name + ":Backbone").c_str(), unknown);

    const std::string &backbone = resolvePath(backbone_path);
    if (backbone_path == unknown) {
      status = loadLayerConfigIni(ini, layer, sec_name);
    } else if (fileIni(backbone_path)) {
      status = loadBackboneConfigIni(ini, backbone, model, sec_name);
      NN_INI_RETURN_STATUS();
      continue;
    } else {
      status = loadBackboneConfigExternal(ini, backbone, layer, sec_name);
    }
    NN_INI_RETURN_STATUS();

    status = model.addLayer(layer);
    NN_INI_RETURN_STATUS();
  }
  ml_logd("parsing ini finished");

  if (model.getFlatGraph().empty()) {
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

  if (model_file_context != nullptr) {
    ml_loge(
      "model_file_context is already initialized, there is a possiblity that "
      "last load from config wasn't finished correctly, and model loader is "
      "reused");
    return ML_ERROR_UNKNOWN;
  }

  model_file_context = std::make_unique<AppContext>();

  auto config_realpath_char = realpath(config.c_str(), nullptr);
  if (config_realpath_char == nullptr) {
    ml_loge("failed to resolve config path to absolute path, reason: %s",
            strerror(errno));
    return ML_ERROR_INVALID_PARAMETER;
  }
  std::string config_realpath(config_realpath_char);
  free(config_realpath_char);

  auto pos = config_realpath.find_last_of("/");
  if (pos == std::string::npos) {
    ml_loge("resolved model path does not contain any path seperater. %s",
            config_realpath.c_str());
    return ML_ERROR_UNKNOWN;
  }

  auto base_path = config_realpath.substr(0, pos);
  model_file_context->setWorkingDirectory(base_path);
  ml_logd("for the current model working directory is set to %s",
          base_path.c_str());

  int status = loadFromConfig(config_realpath, model, false);
  model_file_context.reset();
  return status;
}

/**
 * @brief     load all of model and dataset from given config file
 */
int ModelLoader::loadFromConfig(std::string config, NeuralNetwork &model,
                                bool bare_layers) {
  if (fileIni(config)) {
    return loadFromIni(config, model, bare_layers);
  }

  return ML_ERROR_INVALID_PARAMETER;
}

bool ModelLoader::fileExt(const std::string &filename, const std::string &ext) {
  size_t position = filename.find_last_of(".");
  if (position == std::string::npos)
    return false;

  if (filename.substr(position + 1) == ext) {
    return true;
  }

  return false;
}

bool ModelLoader::fileIni(const std::string &filename) {
  return fileExt(filename, "ini");
}

bool ModelLoader::fileTfLite(const std::string &filename) {
  return fileExt(filename, "tflite");
}

} // namespace nntrainer
