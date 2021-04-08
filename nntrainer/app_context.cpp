// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   app_context.cpp
 * @date   10 November 2020
 * @brief  This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 */
#include <dirent.h>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <iniparser.h>

#include <app_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <util_func.h>

#include <adam.h>
#include <sgd.h>

#include <activation_layer.h>
#include <addition_layer.h>
#include <bn_layer.h>
#include <concat_layer.h>
#include <conv2d_layer.h>
#include <embedding.h>
#include <fc_layer.h>
#include <flatten_layer.h>
#include <input_layer.h>
#include <loss_layer.h>
#include <lstm.h>
#include <nntrainer_error.h>
#include <output_layer.h>
#include <parse_util.h>
#include <plugged_layer.h>
#include <pooling2d_layer.h>
#include <preprocess_flip_layer.h>
#include <preprocess_translate_layer.h>
#include <rnn.h>
#include <time_dist.h>

#ifdef ENABLE_TFLITE_BACKBONE
#include <tflite_layer.h>
#endif

#ifdef ENABLE_NNSTREAMER_BACKBONE
#include <nnstreamer_layer.h>
#endif

/// add #ifdef across platform
static std::string layerlib_suffix = "layer.so";
static const std::string func_tag = "[AppContext] ";

#ifdef NNTRAINER_CONF_PATH
constexpr const char *DEFAULT_CONF_PATH = NNTRAINER_CONF_PATH;
#else
constexpr const char *DEFAULT_CONF_PATH = "/etc/nntrainer.ini";
#endif

constexpr const char *getConfPath() { return DEFAULT_CONF_PATH; }

namespace nntrainer {

namespace {

/**
 * @brief Get the plugin path from conf ini
 *
 * @return std::string plugin path
 */
std::string getPluginPathConf(const std::string &suffix) {
  std::string conf_path{getConfPath()};

  ml_logd("%s conf path: %s", func_tag.c_str(), conf_path.c_str());
  if (!isFileExist(conf_path)) {
    ml_logw(
      "%s conf path does not exist, skip getting plugin path from the conf",
      func_tag.c_str());
    return std::string();
  }

  dictionary *ini = iniparser_load(conf_path.c_str());
  NNTR_THROW_IF(ini == nullptr, std::runtime_error)
    << func_tag << "loading ini failed";

  auto freedict = [ini] { iniparser_freedict(ini); };

  std::string s{"plugins:"};

  s += suffix;

  const char *path = iniparser_getstring(ini, s.c_str(), NULL);
  NNTR_THROW_IF_CLEANUP(path == nullptr, std::invalid_argument, freedict)
    << func_tag << "plugins layer failed";

  freedict();
  std::string ret{path};
  return ret;
}

/**
 * @brief Get the plugin paths
 *
 * @return std::vector<std::string> list of paths to search for
 */
std::vector<std::string> getPluginPaths() {
  std::vector<std::string> ret;

  /*** @note NNTRAINER_PATH is an environment variable stating a @a directory
   * where you would like to look for the layers, while NNTRAINER_CONF_PATH is a
   * (buildtime hardcoded @a file path) to locate configuration file *.ini file
   */
  /*** @note for now, NNTRAINER_PATH is a SINGLE PATH rather than serise of path
   * like PATH environment variable. this could be improved but for now, it is
   * enough
   */
  const char *env_path = std::getenv("NNTRAINER_PATH");
  if (env_path != nullptr) {
    if (isFileExist(env_path)) {
      ml_logd("NNTRAINER_PATH is defined and valid. path: %s", env_path);
      ret.emplace_back(env_path);
    } else {
      ml_logw("NNTRAINER_PATH is given but it is not valid. path: %s",
              env_path);
    }
  }

  std::string conf_path = getPluginPathConf("layer");
  if (conf_path != "") {
    ret.emplace_back(conf_path);
    ml_logd("DEFAULT CONF PATH, path: %s", conf_path.c_str());
  }

  return ret;
}

/**
 * @brief Get the Full Path from given string
 * @details path is resolved in the following order
 * 1) if @a path is absolute, return path
 * ----------------------------------------
 * 2) if @a base == "" && @a path == "", return "."
 * 3) if @a base == "" && @a path != "", return @a path
 * 4) if @a base != "" && @a path == "", return @a base
 * 5) if @a base != "" && @a path != "", return @a base + "/" + path
 *
 * @param path path to calculate from base
 * @param base base path
 * @return const std::string
 */
const std::string getFullPath(const std::string &path,
                              const std::string &base) {
  /// if path is absolute, return path
  if (path[0] == '/') {
    return path;
  }

  if (base == std::string()) {
    return path == std::string() ? "." : path;
  }

  return path == std::string() ? base : base + "/" + path;
}

} // namespace

std::mutex factory_mutex;

/**
 * @brief finialize global context
 *
 */
static void fini_global_context_nntrainer(void) __attribute__((destructor));

static void fini_global_context_nntrainer(void) {}

std::once_flag global_app_context_init_flag;

static void add_default_object(AppContext &ac) {
  /// @note all layers should be added to the app_context to gaurantee that
  /// createLayer/createOptimizer class is created
  using OptType = ml::train::OptimizerType;
  ac.registerFactory(ml::train::createOptimizer<SGD>, SGD::type, OptType::SGD);
  ac.registerFactory(ml::train::createOptimizer<Adam>, Adam::type,
                     OptType::ADAM);
  ac.registerFactory(AppContext::unknownFactory<ml::train::Optimizer>,
                     "unknown", OptType::UNKNOWN);

  using LayerType = ml::train::LayerType;
  ac.registerFactory(nntrainer::createLayer<InputLayer>, InputLayer::type,
                     LayerType::LAYER_IN);
  ac.registerFactory(nntrainer::createLayer<FullyConnectedLayer>,
                     FullyConnectedLayer::type, LayerType::LAYER_FC);
  ac.registerFactory(nntrainer::createLayer<BatchNormalizationLayer>,
                     BatchNormalizationLayer::type, LayerType::LAYER_BN);
  ac.registerFactory(nntrainer::createLayer<Conv2DLayer>, Conv2DLayer::type,
                     LayerType::LAYER_CONV2D);
  ac.registerFactory(nntrainer::createLayer<Pooling2DLayer>,
                     Pooling2DLayer::type, LayerType::LAYER_POOLING2D);
  ac.registerFactory(nntrainer::createLayer<FlattenLayer>, FlattenLayer::type,
                     LayerType::LAYER_FLATTEN);
  ac.registerFactory(nntrainer::createLayer<ActivationLayer>,
                     ActivationLayer::type, LayerType::LAYER_ACTIVATION);
  ac.registerFactory(nntrainer::createLayer<AdditionLayer>, AdditionLayer::type,
                     LayerType::LAYER_ADDITION);
  ac.registerFactory(nntrainer::createLayer<OutputLayer>, OutputLayer::type,
                     LayerType::LAYER_MULTIOUT);
  ac.registerFactory(nntrainer::createLayer<ConcatLayer>, ConcatLayer::type,
                     LayerType::LAYER_CONCAT);
  ac.registerFactory(nntrainer::createLayer<LossLayer>, LossLayer::type,
                     LayerType::LAYER_LOSS);
  ac.registerFactory(nntrainer::createLayer<PreprocessFlipLayer>,
                     PreprocessFlipLayer::type,
                     LayerType::LAYER_PREPROCESS_FLIP);
  ac.registerFactory(nntrainer::createLayer<PreprocessTranslateLayer>,
                     PreprocessTranslateLayer::type,
                     LayerType::LAYER_PREPROCESS_TRANSLATE);
#ifdef ENABLE_NNSTREAMER_BACKBONE
  ac.registerFactory(nntrainer::createLayer<NNStreamerLayer>,
                     NNStreamerLayer::type,
                     LayerType::LAYER_BACKBONE_NNSTREAMER);
#endif
#ifdef ENABLE_TFLITE_BACKBONE
  ac.registerFactory(nntrainer::createLayer<TfLiteLayer>, TfLiteLayer::type,
                     LayerType::LAYER_BACKBONE_TFLITE);
#endif
  ac.registerFactory(nntrainer::createLayer<EmbeddingLayer>,
                     EmbeddingLayer::type, LayerType::LAYER_EMBEDDING);

  ac.registerFactory(nntrainer::createLayer<RNNLayer>, RNNLayer::type,
                     LayerType::LAYER_RNN);

  ac.registerFactory(nntrainer::createLayer<LSTMLayer>, LSTMLayer::type,
                     LayerType::LAYER_LSTM);

  ac.registerFactory(nntrainer::createLayer<TimeDistLayer>, TimeDistLayer::type,
                     LayerType::LAYER_TIME_DIST);

  ac.registerFactory(AppContext::unknownFactory<nntrainer::Layer>, "unknown",
                     LayerType::LAYER_UNKNOWN);
}

static void add_extension_object(AppContext &ac) {
  auto dir_list = getPluginPaths();

  for (auto &path : dir_list) {
    try {
      ac.registerLayerFromDirectory(path);
    } catch (std::exception &e) {
      ml_logw("tried to register extension from %s but failed, reason: %s",
              path.c_str(), e.what());
    }
  }
}

static void registerer(AppContext &ac) noexcept {
  try {
    add_default_object(ac);
    add_extension_object(ac);
  } catch (std::exception &e) {
    ml_loge("registering layers failed!!, reason: %s", e.what());
  } catch (...) {
    ml_loge("registering layer failed due to unknown reason");
  }
};

AppContext &AppContext::Global() {
  static AppContext instance;
  /// in g++ there is a bug that hangs up if caller throws,
  /// so registerer is noexcept although it'd better not
  /// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70298
  std::call_once(global_app_context_init_flag, registerer, std::ref(instance));
  return instance;
}

void AppContext::setWorkingDirectory(const std::string &base) {
  DIR *dir = opendir(base.c_str());

  if (!dir) {
    std::stringstream ss;
    ss << func_tag << "path is not directory or has no permission: " << base;
    throw std::invalid_argument(ss.str().c_str());
  }
  closedir(dir);

  char *ret = realpath(base.c_str(), nullptr);

  if (ret == nullptr) {
    std::stringstream ss;
    ss << func_tag << "failed to get canonical path for the path: ";
    throw std::invalid_argument(ss.str().c_str());
  }

  working_path_base = std::string(ret);
  ml_logd("working path base has set: %s", working_path_base.c_str());
  free(ret);
}

const std::string AppContext::getWorkingPath(const std::string &path) {
  return getFullPath(path, working_path_base);
}

int AppContext::registerLayer(const std::string &library_path,
                              const std::string &base_path) {
  const std::string full_path = getFullPath(library_path, base_path);

  void *handle = dlopen(full_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  const char *error_msg = dlerror();

  NNTR_THROW_IF(handle == nullptr, std::invalid_argument)
    << func_tag << "open plugin failed, reason: " << error_msg;

  nntrainer::LayerPluggable *pluggable =
    reinterpret_cast<nntrainer::LayerPluggable *>(
      dlsym(handle, "ml_train_layer_pluggable"));

  error_msg = dlerror();
  auto close_dl = [handle] { dlclose(handle); };
  NNTR_THROW_IF_CLEANUP(error_msg != nullptr || pluggable == nullptr,
                        std::invalid_argument, close_dl)
    << func_tag << "loading symbol failed, reason: " << error_msg;

  auto layer = pluggable->createfunc();
  NNTR_THROW_IF_CLEANUP(layer == nullptr, std::invalid_argument, close_dl)
    << func_tag << "created pluggable layer is null";
  auto type = layer->getType();
  NNTR_THROW_IF_CLEANUP(type == "", std::invalid_argument, close_dl)
    << func_tag << "custom layer must specify type name, but it is empty";
  pluggable->destroyfunc(layer);

  FactoryType<nntrainer::Layer> factory_func =
    [pluggable](const PropsType &prop) {
      std::unique_ptr<nntrainer::Layer> layer =
        std::make_unique<internal::PluggedLayer>(pluggable);

      return layer;
    };

  return registerFactory<nntrainer::Layer>(factory_func, type);
}

std::vector<int>
AppContext::registerLayerFromDirectory(const std::string &base_path) {
  DIR *dir = opendir(base_path.c_str());

  NNTR_THROW_IF(dir == nullptr, std::invalid_argument)
    << func_tag << "failed to open the directory: " << base_path;

  struct dirent *entry;

  std::vector<int> keys;
  while ((entry = readdir(dir)) != NULL) {
    if (endswith(entry->d_name, layerlib_suffix)) {
      try {
        int key = registerLayer(entry->d_name, base_path);
        keys.emplace_back(key);
      } catch (std::exception &e) {
        closedir(dir);
        throw;
      }
    }
  }

  closedir(dir);

  return keys;
}

} // namespace nntrainer
