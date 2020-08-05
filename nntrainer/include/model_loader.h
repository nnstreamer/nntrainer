// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	model_loader.h
 * @date	5 August 2020
 * @brief	This is model loader class for the Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __MODEL_LOADER_H__
#define __MODEL_LOADER_H__
#ifdef __cplusplus

#include <iniparser.h>
#include <model_loader.h>
#include <neuralnet.h>

namespace nntrainer {

/**
 * @class   ModelLoader
 * @brief   Model Loader class to load model from various config files
 */
class ModelLoader {
public:
  /**
   * @brief     Constructor of the model loader
   */
  ModelLoader() {}

  /**
   * @brief     Destructor of the model loader
   */
  ~ModelLoader() {}

  /**
   * @brief     load all of model and dataset from given config file
   * @param[in] config config file path
   * @param[in/out] model model to be loaded
   */
  int loadFromConfig(std::string config, NeuralNetwork &network);

private:
  /**
   * @brief     load all of model and dataset from ini
   * @param[in] config config file path
   * @param[in/out] model model to be loaded
   */
  int loadFromIni(std::string ini_file, NeuralNetwork &network);

  /**
   * @brief     load dataset config from ini
   * @param[in] ini dictionary containing the config
   * @param[in] model model to be loaded
   */
  int loadDatasetConfigIni(dictionary *ini, NeuralNetwork &network);

  /**
   * @brief     load network config from ini
   * @param[in] ini dictionary containing the config
   * @param[in/out] model model to be loaded
   */
  int loadNetworkConfigIni(dictionary *ini, NeuralNetwork &network);

  /**
   * @brief     load layer config from ini
   * @param[in] ini dictionary containing the config
   * @param[in/out] layer layer to be loaded
   * @param[in] layer_name name of the layer to be loaded
   */
  int loadLayerConfigIni(dictionary *ini, std::shared_ptr<Layer> &layer,
                         std::string layer_name);

  const char *unknown = "Unknown";
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __MODEL_LOADER_H__ */
