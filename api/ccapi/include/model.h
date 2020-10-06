// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	model.h
 * @date	14 October 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is model interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_MODEL_H__
#define __ML_TRAIN_MODEL_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <string>
#include <vector>

#include <dataset.h>
#include <layer.h>
#include <optimizer.h>

/** Define more aliases for the model in the API */
namespace ml {
namespace train {

/**
 * @brief     Enumeration of Network Type
 */
enum class ModelType {
  KNN,        /** k Nearest Neighbor */
  REGRESSION, /** Logistic Regression */
  NEURAL_NET, /** Neural Network */
  UNKNOWN     /** Unknown */
};

/**
 * @class   Model Class
 * @brief   Model Class containing configuration, layers, optimizer and dataset
 */
class Model {
public:
  /**
   * @brief     Destructor of Model Class
   */
  virtual ~Model() = default;

  /**
   * @brief     Get Loss from the previous ran batch of data
   * @retval    loss value
   */
  virtual float getLoss() = 0;

  /**
   * @brief     Get Loss from the previous epoch of training data
   * @retval    loss value
   */
  virtual float getTrainingLoss() = 0;

  /**
   * @brief     Get Loss from the previous epoch of validation data
   * @retval    loss value
   */
  virtual float getValidationLoss() = 0;

  /**
   * @brief     Get Learning rate
   * @retval    Learning rate
   */
  virtual float getLearningRate() = 0;

  /**
   * @brief     Create and load the Network with ini configuration file.
   * @param[in] config config file path
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int loadFromConfig(std::string config) = 0;

  /**
   * @brief     set Property of Network
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setProperty(std::vector<std::string> values) = 0;

  /**
   * @brief     Initialize Network. This should be called after set all
   * hyperparameters.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int init() = 0;

  /**
   * @brief     save model and training parameters into file
   */
  virtual void saveModel() = 0;

  /**
   * @brief     read model and training parameters from file
   */
  virtual void readModel() = 0;

  /**
   * @brief     get Epochs
   * @retval    epochs
   */
  virtual unsigned int getEpochs() = 0;

  /**
   * @brief     Run Model train
   * @param[in] values hyper parameters
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int train(std::vector<std::string> values = {}) = 0;

  /**
   * @brief     Run Model train with callback function by user
   * @param[in] dataset set the dataset
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setDataset(std::shared_ptr<Dataset> dataset) = 0;

  /**
   * @brief     add layer into neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int addLayer(std::shared_ptr<Layer> layer) = 0;

  /**
   * @brief     set optimizer for the neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setOptimizer(std::shared_ptr<Optimizer> optimizer) = 0;

  /*
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int getLayer(const char *name, std::shared_ptr<Layer> *layer) = 0;

  /**
   * @brief     Property Enumeration
   */
  enum class PropertyType {
    loss = 0,
    loss_type = 1,
    batch_size = 2,
    epochs = 3,
    save_path = 4,
    continue_train = 5,
    unknown = 6
  };

  /**
   * @brief Print Option when printing model info. The function delegates to the
   * `print`
   * @param out std::ostream to print
   * @param preset preset from `ml_train_summary_type_e`
   */
  virtual void printPreset(std::ostream &out, unsigned int preset) = 0;
};

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Model>
createModel(ModelType type, const std::vector<std::string> &properties = {});

} // namespace train
} // namespace ml

#else
#error "CPP versions c++14 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_MODEL_H__
