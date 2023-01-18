/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer tensor_trainer subplugin for nntrainer
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file   tensor_trainer_nntrainer.hh
 * @date   02 Dec 2022
 * @brief  NNStreamer tensor_trainer subplugin header
 * @see    http://github.com/nnstreamer/nntrainer
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Hyunil Park <hyunil46.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <condition_variable>
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
  bool is_training_complete;

  int64_t tensors_inputsize[NNS_TENSOR_SIZE_LIMIT];
  int64_t num_tensors; /**< The number of tensors in the received a sample */
  int64_t num_inputs; /**< The number of tensors used as input in the received a
                         sample */
  int64_t num_labels; /**< The number of tensors used as label in the received a
                         sample */
  int64_t num_training_samples; /**< The number of training samples to be taken
                                   for training model */
  int64_t num_validation_samples; /**< The number of validation samples to be
                                     taken for validation model */
  int64_t total_num_samples; /**< Total number of samples received for creating
                                model */
  int64_t num_epochs;        /**< The number of epoch */
  int64_t num_push_data;     /**< The number of samples pushed by
                                NNStreamer(tensor_trainer)*/
  std::string model_config;
  std::string model_save_path; /**< Model is finally stored */

  GCond *training_complete_cond;

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
   * @param _total_num_samples Total number of samples received for creating
   * model
   * @param _num_inputs The number of tensors used as input in the received a
   * sample
   * @param _num_labels The number of tensors used as label in the received a
   * sample
   * @param _tensors_size[] size of each tensor in a sample
   */
  InputTensorsInfo(int64_t _total_num_samples, int64_t _num_inputs,
                   int64_t _num_labels, int64_t _tensors_size[]);

  /**
   * @brief Destroy the InputTensorsInfo object
   */
  ~InputTensorsInfo();

  bool is_data_wait_locked;
  bool is_data_full_locked;
  unsigned int queue_size;
  unsigned int queue_front;
  unsigned int queue_rear;
  unsigned int queue_count;
  int64_t push_count; /**< The number of samples pushed to queue by
                         NNStreamer(tensor_trainer) */
  int64_t pop_count;  /**< The number of pop from the queue for pushing samples
                         to nntrainer */
  int64_t input_size[NNS_TENSOR_SIZE_LIMIT]; // feature size * data type
  int64_t label_size[NNS_TENSOR_SIZE_LIMIT];
  int64_t total_num_samples; /**< Total number of samples received for creating
                                model */
  int64_t num_inputs; /**< The number of tensors in the received a sample */
  int64_t num_labels; /**< The number of tensors used as label in the received a
                         sample */

  std::vector<TensorData>
    tensor_data; /**< Manage multiple inputs and labels data */

  std::mutex data_wait_lock;
  std::mutex data_full_lock;
  std::condition_variable data_wait;
  std::condition_variable data_full;
};
} // namespace NNTrainer
