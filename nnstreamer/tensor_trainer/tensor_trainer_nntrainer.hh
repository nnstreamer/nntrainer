/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer tensor_trainer subplugin for nntrainer
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file   tensor_trainer_nntrainer.hh
 * @date   07 November 2022
 * @brief  NNStreamer tensor_trainer subplugin header
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Hyunil Park <hyunil46.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <model.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>
#include <vector>

namespace NNTrainer {

class InputTensorsInfo;

/**
 * @brief NNTrainer interface for nnstreamer trainer subplugin
 */
class NNTrainerTrain {
public:
  /**
   * @brief Construct a new NNTrainerTrain object
   */
  NNTrainerTrain(const std::string &_model_config);

  /**
   * @brief Destroy the NNTrainerTrain object
   * @param _model_config model configuration file path
   */
  ~NNTrainerTrain() = default;

  /**
   * @brief Set the Batch Size
   * @param _batch_size batch size
   */
  void setBatchSize(unsigned int _batch_size) { batch_size = _batch_size; }

  /**
   * @brief Set the number of train data
   * @param _num_of_train_data number of train data
   */
  void setNumberOfTrainData(unsigned int _num_of_train_data) {
    num_of_train_data = _num_of_train_data;
  }

  /**
   * @brief Set the number of validation data
   * @param _num_of_valid_data number of validation data
   */
  void setNumberOfValidData(unsigned int _num_of_valid_data) {
    num_of_valid_data = _num_of_valid_data;
  }

  /**
   * @brief Set the number of label list
   * @param _num_of_label_list number of label list
   */
  void setNumberOfLabelList(unsigned int _num_of_label_list) {
    num_of_label_list = _num_of_label_list;
  }

  /**
   * @brief Set the number of input list
   * @param _num_of_label_list number of input list
   */
  void setNumberOfInputList(unsigned int _num_of_input_list) {
    num_of_input_list = _num_of_input_list;
  }

  /**
   * @brief create model
   */
  void createModel();

  /**
   * @brief train model
   */
  void trainModel();

  /**
   * @brief send input data to NNTrainerTrain
   * @param input input tensor memory imported input and label
   */
  void sendTensorData(const GstTensorMemory *input);

  std::unique_ptr<NNTrainer::InputTensorsInfo> train_data, valid_data;
  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_valid;
  float training_loss, validation_loss;

private:
  unsigned int batch_size;
  unsigned int num_of_train_data;
  unsigned int num_of_valid_data;
  unsigned int num_of_label_list;
  unsigned int num_of_input_list;

  std::string model_config;
  std::unique_ptr<ml::train::Model> model;
};

/**
 * @brief Input tensors data and information
 */
class InputTensorsInfo {
public:
  InputTensorsInfo(unsigned int num_of_samples, unsigned int num_of_inputs,
                   unsigned num_of_labels);
  unsigned int count;
  unsigned int num_of_samples;
  unsigned int num_of_inputs;
  unsigned int num_of_labels;
  std::vector<float *> inputs;
  std::vector<float *> labels;
};

InputTensorsInfo::InputTensorsInfo(unsigned int _num_of_samples,
                                   unsigned int _num_of_inputs,
                                   unsigned int _num_of_labels) :
  count(0),
  num_of_samples(_num_of_samples),
  num_of_inputs(_num_of_inputs),
  num_of_labels(_num_of_labels) {
  inputs.reserve(num_of_samples * num_of_inputs);
  labels.reserve(num_of_samples * num_of_labels);
}
} // namespace NNTrainer
