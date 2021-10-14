// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   model.h
 * @date   14 October 2020
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is model interface for c++ API
 *
 * @note This is experimental API and not stable.
 */

#ifndef __ML_TRAIN_MODEL_H__
#define __ML_TRAIN_MODEL_H__

#if __cplusplus >= MIN_CPP_VERSION

#include <string>
#include <type_traits>
#include <vector>

#include <nntrainer-api-common.h>

#include <dataset.h>
#include <layer.h>
#include <optimizer.h>
#include <tensor_dim.h>

/** Define more aliases for the model in the API */
namespace ml {
namespace train {

/**
 * @brief     Enumeration of Network Type
 */
enum class ModelType {
  KNN,        /** k Nearest Neighbor */
  NEURAL_NET, /** Neural Network */
  UNKNOWN     /** Unknown */
};

/**
 * @brief     Enumeration to state type of reference of layers
 */
enum class ReferenceLayersType {
  BACKBONE,  /** backbone */
  RECURRENT, /** recurrent */
};

/**
 * @brief Model saving options
 *
 */
enum class ModelFormat {
  MODEL_FORMAT_BIN =
    ML_TRAIN_MODEL_FORMAT_BIN, /**< raw bin file saves model weights required
for inference and training without any configurations*/
  MODEL_FORMAT_INI = ML_TRAIN_MODEL_FORMAT_INI, /**< ini file */
  MODEL_FORMAT_INI_WITH_BIN =
    ML_TRAIN_MODEL_FORMAT_INI_WITH_BIN, /**< ini file with save_path defined
                                           where the binaray will be saved */
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
   * @brief     Create and load the Network with ini configuration file.
   * @param[in] config config file path
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int loadFromConfig(const std::string &config) = 0;

  /**
   * @brief     Minimal set of properties that must be supported by the model
   * @details   The minimal properies:
                - loss_type
                - batch_size
                - epochs
                - save_path
                - continue_train
   */
  /**
   * @brief     set Property of Network
   * @param[in] values values of property
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual void setProperty(const std::vector<std::string> &values) = 0;

  /**
   * @brief     Compile Network. This should be called before initialize
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int compile() = 0;

  /**
   * @brief     Initialize Network. This should be called after setting the
   * property and compiling.
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int initialize() = 0;

  /**
   * @brief  load model states and training parameters from a file
   * @param file_path file_path to save the model, if full path is not
   * given, it should be saved inside working directory
   * @param format format to save parameters
   */
  virtual void save(const std::string &file_path,
                    ModelFormat format = ModelFormat::MODEL_FORMAT_BIN) = 0;

  /**
   * @brief  load model with regard to the format
   * @param file_path file_path to save the model, if full path is not
   * given, it should be saved inside working directory
   * @param format format to save parameters
   */
  virtual void load(const std::string &file_path,
                    ModelFormat format = ModelFormat::MODEL_FORMAT_BIN) = 0;

  /**
   * @brief     Run Model training and validation
   * @param[in] values hyper parameters
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   * @details   This function accepts vector of properties in the format -
   *  { std::string property_name, void * property_val, ...}
   */
  virtual int train(const std::vector<std::string> &values = {}) = 0;

  /**
   * @brief     Run Model train with callback function by user
   * @param[in] mode mode of the dataset
   * @param[in] dataset set the dataset
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setDataset(const ml::train::DatasetModeType &mode,
                         std::shared_ptr<Dataset> dataset) = 0;

  /**
   * @brief     add layer into neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int addLayer(std::shared_ptr<Layer> layer) = 0;

  /**
   * @brief add refering to reference layers.
   * @note This method does add the provided layers itself but adds a deep copy
   * of the passed layers to the model. The layers passed to this function can
   * be reused later.
   *
   * @param reference a group of layers being referred to.
   * @param type type of reference layers
   * @param scope scope of added layers, identifier will be added
   * @param input_layers input layers which will be used to connect input layers
   * @param start_layers start layers which will be used to specify start of the
   * layers inside @a reference
   * @param end_layers end layers which will be used to specify end of the
   * layers inside @a reference
   * @param type_properties type dependent properties
   */
  virtual void addWithReferenceLayers(
    const std::vector<std::shared_ptr<Layer>> &reference,
    const std::string &scope, const std::vector<std::string> &input_layers,
    const std::vector<std::string> &start_layers,
    const std::vector<std::string> &end_layers, ReferenceLayersType type,
    const std::vector<std::string> &type_properties = {}) = 0;

  /**
   * @brief     set optimizer for the neural network model
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int setOptimizer(std::shared_ptr<Optimizer> optimizer) = 0;

  /**
   * @brief     get layer by name from neural network model
   * @param[in] name name of the layer to get
   * @param[out] layer shared_ptr to hold the layer to get
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  virtual int getLayer(const char *name, std::shared_ptr<Layer> *layer) = 0;

  /**
   * @brief     get input dimension of a model
   * @retval    std::vector<ml::train::TensorDim> input dimension
   */
  virtual std::vector<ml::train::TensorDim> getInputDimension() = 0;

  /**
   * @brief     get output dimension of a model
   * @retval    std::vector<ml::train::TensorDim> output dimension
   */
  virtual std::vector<ml::train::TensorDim> getOutputDimension() = 0;

  /**
   * @brief     Run the inference of the model
   * @param[in] batch batch size of current input
   * @param[in] input inputs as a list of each input data
   * @param[in] label labels as a list of each label data
   * @retval list of output as float *
   * @note The output memory must not be freed by the caller
   */
  virtual std::vector<float *> inference(unsigned int batch,
                                         std::vector<float *> &input,
                                         std::vector<float *> &label) = 0;

  /**
   * @brief     Summarize the model
   * @param out std::ostream to get the model summary
   * @param verbosity verbosity of the summary
   */
  virtual void summarize(std::ostream &out,
                         ml_train_summary_type_e verbosity) = 0;

  /**
   * @brief     Get Loss
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
};

/**
 * @brief Factory creator with constructor for optimizer
 */
std::unique_ptr<Model>
createModel(ModelType type, const std::vector<std::string> &properties = {});

} // namespace train
} // namespace ml

#else
#error "CPP versions c++17 or over are only supported"
#endif // __cpluscplus
#endif // __ML_TRAIN_MODEL_H__
