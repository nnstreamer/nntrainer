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
 * @brief NNTrainer interface for nnstreamer trainer sub-plugin
 */
class NNTrainerTrain {
public:
  /**
   * @brief Construct a new NNTrainerTrain object
   * @param prop tensor trainer sub-plugin properties
   * @param _model_config model configuration file path
   */
  NNTrainerTrain(const GstTensorTrainerProperties *prop,
                 const std::string &_model_config);

  /**
   * @brief Destroy the NNTrainerTrain object
   */
  ~NNTrainerTrain() = default;

  NNTrainerTrain *GetNNTrainerTrain() { return this; }

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
   * @brief Get run stats
   */
  void getRunStats();

  /**
   * @brief Get NNStreamer tensor_trainer properties
   * @param prop Tensor trainer sub-plugin properties
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

  unsigned int tensors_inputsize[NNS_TENSOR_SIZE_LIMIT];
  unsigned int
    num_tensors; /**< The number of tensors in the received a sample */
  unsigned int num_inputs; /**< The number of tensors used as input in the
                              received a sample */
  unsigned int num_labels; /**< The number of tensors used as label in the
                              received a sample */
  unsigned int num_training_samples;   /**< The number of training samples to be
                                          taken for training model */
  unsigned int num_validation_samples; /**< The number of validation samples to
                                          be taken for validation model */
  unsigned int total_num_samples;      /**< Total number of samples received for
                                          creating model */
  unsigned int num_epochs;             /**< The number of epoch */
  unsigned int num_push_data;          /**< The number of samples pushed by
                                          NNStreamer(tensor_trainer) */
  std::string model_config;
  std::string model_save_path; /**< Model is finally stored */
  std::string model_load_path; /**< Path to load an exisiting model to use for
                                  training a new model */

  ml::train::RunStats train_stats;
  ml::train::RunStats valid_stats;

  GstTensorTrainerEventNotifier *notifier; /**< a handle of event notify */
  bool stop_model_training;

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
  InputTensorsInfo(unsigned int _total_num_samples, unsigned int _num_inputs,
                   unsigned int _num_labels, unsigned int _tensors_size[]);

  /**
   * @brief Destroy the InputTensorsInfo object
   */
  ~InputTensorsInfo();

  bool is_data_wait_locked;
  bool is_data_full_locked;
  unsigned int queue_size;
  unsigned int queue_front;
  unsigned int queue_rear;
  unsigned int queue_count; /**< The number of data in queue */
  unsigned int push_count;  /**< The number of samples pushed to queue by
                               NNStreamer(tensor_trainer) */
  unsigned int pop_count;   /**< The number of pop from the queue for pushing
                               samples to nntrainer */
  unsigned int
    input_size[NNS_TENSOR_SIZE_LIMIT]; /**< feature size * data type */
  unsigned int label_size[NNS_TENSOR_SIZE_LIMIT];
  unsigned int total_num_samples; /**< Total number of samples received for
                                     training model */
  unsigned int
    num_inputs; /**< The number of tensors in the received a sample */
  unsigned int num_labels; /**< The number of tensors used as label in the
                              received a sample */

  std::vector<TensorData>
    tensor_data; /**< Manage multiple inputs and labels data */

  std::mutex data_wait_lock;
  std::mutex data_full_lock;
  std::condition_variable data_wait;
  std::condition_variable data_full;

  /**
   * @brief get sample data
   *
   * @param input input data
   * @param label label data
   * @param last set TRUE if data is last
   */
  void getSample(float **input, float **label, bool *last);
};
} // namespace NNTrainer
