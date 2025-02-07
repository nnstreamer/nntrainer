// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   model_loader.h
 * @date   5 August 2020
 * @brief  This is model loader class for the Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __MODEL_LOADER_H__
#define __MODEL_LOADER_H__
#ifdef __cplusplus

#include <memory>

#include <engine.h>
#include <iniparser.h>
#include <neuralnet.h>

namespace nntrainer {

class OptimizerWrapped;

/**
 * @class   ModelLoader
 * @brief   Model Loader class to load model from various config files
 */
class ModelLoader {
public:
  /**
   * @brief     Constructor of the model loader
   */
  ModelLoader(const Engine &ct_eng_ = Engine::Global()) :
    ct_engine(ct_eng_), model_file_engine(nullptr) {}

  /**
   * @brief     Destructor of the model loader
   */
  ~ModelLoader() {}

  /**
   * @brief     load all properties from context
   * @param[in/out] model model to be loaded
   */
  int loadFromContext(NeuralNetwork &model);

  /**
   * @brief     load all of model and dataset from given config file
   * @param[in] config config file path
   * @param[in/out] model model to be loaded
   */
  int loadFromConfig(std::string config, NeuralNetwork &model);

private:
  /**
   * @brief     load all of model from given config file
   * @param[in] config config file path
   * @param[in/out] model model to be loaded
   * @param[in] bare_layers load only the layers as backbone if enabled
   */
  int loadFromConfig(std::string config, NeuralNetwork &model,
                     bool bare_layers);

  /**
   * @brief     load all of model and dataset from ini
   * @param[in] ini_file config file path
   * @param[in/out] model model to be loaded
   */
  int loadFromIni(std::string ini_file, NeuralNetwork &model, bool bare_layers);

  /**
   * @brief     load dataset config from ini
   * @param[in] ini dictionary containing the config
   * @param[in] model model to be loaded
   */
  int loadDatasetConfigIni(dictionary *ini, NeuralNetwork &model);

  /**
   * @brief     load model config from ini
   * @param[in] ini dictionary containing the config
   * @param[in/out] model model to be loaded
   */
  int loadModelConfigIni(dictionary *ini, NeuralNetwork &model);

  /**
   * @brief     load optimizer config from ini
   * @param[in] ini dictionary containing the config
   * @param[in/out] model model to be loaded
   */
  int loadOptimizerConfigIni(dictionary *ini, NeuralNetwork &model);

  /**
   * @brief     load learning rate scheduler config from ini
   * @param[in] ini dictionary containing the config
   * @param[in/out] optimizer to contain the lr scheduler
   */
  int loadLearningRateSchedulerConfigIni(
    dictionary *ini, std::shared_ptr<ml::train::Optimizer> &optimizer);

  /**
   * @brief     Check if the file extension is the given @a ext
   * @param[in] filename full name of the file
   * @param[in] ext extension to match with
   * @retval true if @a ext, else false
   */
  static bool fileExt(const std::string &filename, const std::string &ext);

  /**
   * @brief     Check if the file extension is ini
   * @param[in] filename full name of the file
   * @retval true if ini, else false
   */
  static bool fileIni(const std::string &filename);

  /**
   * @brief     Check if the file extension is tflite
   * @param[in] filename full name of the file
   * @retval true if tflite, else false
   */
  static bool fileTfLite(const std::string &filename);

  /**
   * @brief resolvePath to absolute path written in a model description
   *
   * @note  if path is absolute path, return path.
   *        if app_context has working directory set, resolve from app_context
   *        if not, resolve path assuming model_path is the current directory.
   *        The behavior relies on the semantics of getWorkingPath();
   * @param path path to resolve
   * @return const std::string resolved path.
   */
  const std::string resolvePath(const std::string &path) {
    auto path_ = ct_engine.getWorkingPath(path);
    return model_file_engine->getWorkingPath(path_);
  }

  /**
   * @brief parse all the properties for a given section
   * @param[in] ini dictionary containing the config
   * @param[in] section_name name of the section for properties to parse
   * @param[in] filter_props the properties to be filtered out
   */
  std::vector<std::string>
  parseProperties(dictionary *ini, const std::string &section_name,
                  const std::vector<std::string> &filter_props = {});

  const char *unknown = "Unknown";
  const char *none = "none";

  Engine ct_engine;
  std::unique_ptr<Engine> model_file_engine;
  /**< model_file specific context which is
  //          referred to as if app_context cannot
  //          resolve some given configuration */
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __MODEL_LOADER_H__ */
