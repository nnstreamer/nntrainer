/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer tensor_trainer subplugin for nntrainer
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file   tensor_trainer_nntrainer.hh
 * @date   02 Dec 2022
 * @brief  NNStreamer tensor_trainer subplugin header
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Hyunil Park <hyunil46.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <model.h>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_trainer.h>
#include <vector>

namespace NNTrainer {

/**
 * @brief Manage multiple inputs and labels data
 */
struct TensorData {
  std::vector<char *> inputs;
  std::vector<char *> labels;
};

class InputTensorsInfo;

/**
 * @brief NNTrainer interface for nnstreamer trainer subplugin
 */
class NNTrainerTrain {
public:
  /**
   * @brief Construct a new NNTrainerTrain object
   * @param prop tensor trainer subplugin properties
   * @param _model_config model configuration file path
   */
  NNTrainerTrain(const GstTensorTrainerProperties *prop,
                 const std::string &_model_config);

  /**
   * @brief Destroy the NNTrainerTrain object
   */
  ~NNTrainerTrain() = default;

  /**
   * @brief Create model
   */
  void createModel();

  /**
   * @brief Train model
   */
  void trainModel();

  /**
   * @brief Create dataset
   */
  void createDataset();

  /**
   * @brief Get NNStreamer tensor_trainer properties
   * @param prop Tensor trainer subplugin properties
   */
  void getNNStreamerProperties(const GstTensorTrainerProperties *prop);

  /**
   * @brief Manage sample data
   */
  std::unique_ptr<NNTrainer::InputTensorsInfo> train_data, valid_data;
  /**
   * @brief Nntrainer dataset
   */
  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_valid;
  float training_loss, validation_loss;

  int64_t tensors_inputsize[NNS_TENSOR_SIZE_LIMIT];
  int64_t num_tensors;
  int64_t num_inputs;
  int64_t num_labels;
  int64_t num_train_samples;
  int64_t num_valid_samples;
  std::string model_config;
  std::string model_save_path;

  GCond *train_complete_cond;

private:
  std::unique_ptr<ml::train::Model> model;
};

/**
 * @brief Manage input tensors data and information
 */
class InputTensorsInfo {
public:
  /**
   * @brief Construct a new InputTensorsInfo object
   * @param _num_samples number of samples
   * @param _num_inputs number of inputs
   * @param _num_labels number of labels
   * @param _tensors_inputsize[] input tensors size
   */
  InputTensorsInfo(int64_t _num_samples, int64_t _num_inputs,
                   int64_t _num_labels, int64_t _tensors_inputsize[]);

  /**
   * @brief Destroy the InputTensorsInfo object
   */
  ~InputTensorsInfo();

  bool is_mutex_locked;
  int64_t push_count;
  int64_t pop_count;
  int64_t input_size[NNS_TENSOR_SIZE_LIMIT]; // feature size * data type
  int64_t label_size[NNS_TENSOR_SIZE_LIMIT];
  int64_t num_samples;
  int64_t num_inputs;
  int64_t num_labels;

  std::vector<TensorData> tensor_data;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
};
} // namespace NNTrainer
