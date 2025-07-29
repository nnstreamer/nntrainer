// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file  model_loader.cpp
 * @date  5 August 2020
 * @brief This is model loader class for the Neural Network
 * @see   https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @author  Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug   No known bugs except for NYI items
 *
 */
#include <sstream>

#include <adam.h>
#include <databuffer_factory.h>
#include <ini_interpreter.h>
#include <model_loader.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <optimizer_wrapped.h>
// #include <time_dist.h>
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

int ModelLoader::loadLearningRateSchedulerConfigIni(
  dictionary *ini, std::shared_ptr<ml::train::Optimizer> &optimizer) {
  int status = ML_ERROR_NONE;

  if (iniparser_find_entry(ini, "LearningRateScheduler") == 0) {
    return ML_ERROR_NONE;
  }

  /** Default to adam optimizer */
  const char *lrs_type =
    iniparser_getstring(ini, "LearningRateScheduler:Type", "unknown");
  std::vector<std::string> properties =
    parseProperties(ini, "LearningRateScheduler", {"type"});

  try {
    auto lrs = app_context.createObject<ml::train::LearningRateScheduler>(
      lrs_type, properties);
    auto opt_wrapped = std::static_pointer_cast<OptimizerWrapped>(optimizer);
    opt_wrapped->setLearningRateScheduler(std::move(lrs));
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("Creating the optimizer failed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

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
      createOptimizerWrapped(opt_type, properties);
    model.setOptimizer(optimizer);
    loadLearningRateSchedulerConfigIni(ini, optimizer);
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

  std::vector<std::string> properties = parseProperties(
    ini, "Model",
    {"optimizer", "learning_rate", "decay_steps", "decay_rate", "beta1",
     "beta2", "epsilon", "type", "save_path", "tensor_type", "tensor_format"});
  try {
    model.setProperty(properties);
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("Creating the optimizer failed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  /** handle save_path as a special case for model_file_context */
  const std::string &save_path =
    iniparser_getstring(ini, "Model:Save_path", unknown);
  if (save_path != unknown) {
    model.setProperty({"save_path=" + resolvePath(save_path)});
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
    std::shared_ptr<ml::train::Optimizer> optimizer =
      createOptimizerWrapped(opt_type, {});
    model.setOptimizer(optimizer);
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("Creating the optimizer failed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::vector<std::string> optimizer_prop = {};

  /** push only if ini_key exist as prop_key=ini_value */
  auto maybe_push = [ini](std::vector<std::string> &prop_vector,
                          const std::string &ini_key,
                          const std::string &prop_key) {
    constexpr const char *LOCAL_UNKNOWN = "unknown";
    std::string ini_value =
      iniparser_getstring(ini, ini_key.c_str(), LOCAL_UNKNOWN);
    if (!istrequal(ini_value, LOCAL_UNKNOWN)) {
      prop_vector.push_back(prop_key + "=" + ini_value);
    }
  };

  const std::vector<std::string> deprecated_optimizer_keys = {
    "learning_rate", "decay_rate", "decay_steps", "beta1", "beta2", "epsilon"};
  for (const auto &key : deprecated_optimizer_keys) {
    maybe_push(optimizer_prop, "Model:" + key, key);
  }

  try {
    model.opt->setProperty(optimizer_prop);
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (...) {
    ml_loge("Settings properties to optimizer failed.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  return status;
}

/**
 * @brief     load dataset config from ini
 */
int ModelLoader::loadDatasetConfigIni(dictionary *ini, NeuralNetwork &model) {
  /************ helper functors **************/
  auto try_parse_datasetsection_for_backward_compatibility = [&]() -> int {
    int status = ML_ERROR_NONE;
    if (iniparser_find_entry(ini, "Dataset") == 0) {
      return ML_ERROR_NONE;
    }

    ml_logw("Using dataset section is deprecated, please consider using "
            "train_set, valid_set, test_set sections");

    /// @note DataSet:BufferSize is parsed for backward compatibility
    std::string bufsizepros("buffer_size=");
    bufsizepros +=
      iniparser_getstring(ini, "DataSet:BufferSize",
                          iniparser_getstring(ini, "DataSet:buffer_size", "1"));

    auto parse_and_set = [&](const char *key, DatasetModeType dt,
                             bool required) -> int {
      const char *path = iniparser_getstring(ini, key, NULL);

      if (path == NULL) {
        return required ? ML_ERROR_INVALID_PARAMETER : ML_ERROR_NONE;
      }

      try {
        model.data_buffers[static_cast<int>(dt)] =
          createDataBuffer(DatasetType::FILE, resolvePath(path).c_str());
        model.data_buffers[static_cast<int>(dt)]->setProperty({bufsizepros});
      } catch (...) {
        ml_loge("path is not valid, path: %s", resolvePath(path).c_str());
        return ML_ERROR_INVALID_PARAMETER;
      }

      return ML_ERROR_NONE;
    };

    status =
      parse_and_set("DataSet:TrainData", DatasetModeType::MODE_TRAIN, true);
    NN_RETURN_STATUS();
    status =
      parse_and_set("DataSet:ValidData", DatasetModeType::MODE_VALID, false);
    NN_RETURN_STATUS();
    status =
      parse_and_set("DataSet:TestData", DatasetModeType::MODE_TEST, false);
    NN_RETURN_STATUS();
    const char *path = iniparser_getstring(ini, "Dataset:LabelData", NULL);
    if (path != NULL) {
      ml_logi("setting labelData is deprecated!, it is essentially noop now!");
    }

    ml_logd("parsing dataset done");
    return status;
  };

  auto parse_buffer_section = [ini, this,
                               &model](const std::string &section_name,
                                       DatasetModeType type) -> int {
    if (iniparser_find_entry(ini, section_name.c_str()) == 0) {
      return ML_ERROR_NONE;
    }
    const char *db_type =
      iniparser_getstring(ini, (section_name + ":type").c_str(), unknown);
    auto &db = model.data_buffers[static_cast<int>(type)];

    /// @todo delegate this to app context (currently there is only file
    /// databuffer so file is directly used)
    if (!istrequal(db_type, "file")) {
      ml_loge("databuffer type is unknonw, type: %s", db_type);
      return ML_ERROR_INVALID_PARAMETER;
    }

    try {
      db = createDataBuffer(DatasetType::FILE);
      const std::vector<std::string> properties =
        parseProperties(ini, section_name, {"type"});

      db->setProperty(properties);
    } catch (std::exception &e) {
      ml_loge("error while creating and setting dataset, %s", e.what());
      return ML_ERROR_INVALID_PARAMETER;
    }

    return ML_ERROR_NONE;
  };

  /************ start of the procedure **************/
  int status = ML_ERROR_NONE;
  status = try_parse_datasetsection_for_backward_compatibility();
  NN_RETURN_STATUS();

  status = parse_buffer_section("train_set", DatasetModeType::MODE_TRAIN);
  NN_RETURN_STATUS();
  status = parse_buffer_section("valid_set", DatasetModeType::MODE_VALID);
  NN_RETURN_STATUS();
  status = parse_buffer_section("test_set", DatasetModeType::MODE_TEST);
  NN_RETURN_STATUS();

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

/**
 * @brief     load all of model and dataset from ini
 */
int ModelLoader::loadFromIni(std::string ini_file, NeuralNetwork &model,
                             bool bare_layers) {
  int status = ML_ERROR_NONE;
  int num_ini_sec = 0;
  dictionary *ini;

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

  auto path_resolver = [this](const std::string &path) {
    return resolvePath(path);
  };

  ml_logd("parsing graph started");
  try {
    std::unique_ptr<GraphInterpreter> ini_interpreter =
      std::make_unique<nntrainer::IniGraphInterpreter>(app_context,
                                                       path_resolver);
    auto graph_representation = ini_interpreter->deserialize(ini_file);

    for (auto &node : graph_representation) {
      model.addLayer(node);
    }
    ml_logd("parsing graph finished");

    if (model.empty()) {
      ml_loge("there is no layer section in the ini file");
      status = ML_ERROR_INVALID_PARAMETER;
    }
  } catch (std::exception &e) {
    ml_loge("failed to load graph, reason: %s ", e.what());
    status = ML_ERROR_INVALID_PARAMETER;
  }

  iniparser_freedict(ini);
  return status;
}

/**
 * @brief     load all properties from context
 */
int ModelLoader::loadFromContext(NeuralNetwork &model) {
  auto props = app_context.getProperties();
  model.setTrainConfig(props);

  return ML_ERROR_NONE;
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

  auto config_realpath_char = getRealpath(config.c_str(), nullptr);
  if (config_realpath_char == nullptr) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    ml_loge("failed to resolve config path to absolute path, reason: %s",
            strerror_r(errno, error_buf, error_buflen));
    return ML_ERROR_INVALID_PARAMETER;
  }
  std::string config_realpath(config_realpath_char);
  free(config_realpath_char);

  auto pos = config_realpath.find_last_of("/");
  if (pos == std::string::npos) {
    ml_loge("resolved model path does not contain any path separator. %s",
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
