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
#include <mutex>
#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_trainer.h>
#include <vector>

/**
 * @brief nnstreamer tensor_trainer sub plugin
 */
namespace NNTrainer {

/**
 * @brief Manage multiple inputs and labels tensors data
 */
struct TensorsData {
  std::vector<char *> inputs;
  std::vector<char *> labels;
};

class TensorsQueue;

/**
 * @brief nntrainer Implementation class for nnstreamer trainer sub-plugin
 */
class NNTrainerImpl {
public:
  /**
   * @brief Construct a new NNTrainerImpl object
   * @param prop tensor trainer sub-plugin properties
   */
  NNTrainerImpl(const GstTensorTrainerProperties *prop);

  /**
   * @brief Destroy the NNTrainerImpl object
   */
  ~NNTrainerImpl() = default;

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
   * @brief Set NNStreamer tensor_trainer properties to member variable
   * @param prop Tensor trainer sub-plugin properties
   */
  void setNNStreamerProperties(const GstTensorTrainerProperties *prop);

  /**
   * @brief Get number of training samples
   */
  unsigned int getNumOfTrainingSamples() { return num_training_samples; }

  /**
   * @brief Get number of validation samples
   */
  unsigned int getNumOfValidationSamples() { return num_validation_samples; }

  /**
   * @brief Get number of total samples
   */
  unsigned int getNumOfTotalSamples() { return total_num_samples; }

  /**
   * @brief Get number of pushed samples
   */
  unsigned int getNumOfPushedSamples() { return num_pushed_samples; }

  /**
   * @brief Increment number of pushed samples
   */
  void incrementNumOfPushedSamples() { ++num_pushed_samples; }

  /**
   * @brief Setting whether to end model training
   */
  void setStopModelTraining(bool value) { stop_model_training = value; }

  /**
   * @brief get Event Notifier
   */
  GstTensorTrainerEventNotifier *getEventNotifier() { return notifier; }

  /**
   * @brief set Event Notifier
   */
  void setEventNotifier(GstTensorTrainerEventNotifier *value) {
    notifier = value;
  }

  /**
   * @brief set Event Notifier version
   */
  void setEventNotifierVersion(uint64_t version) {
    notifier->version = version;
  }

  /**
   * @brief Manage sample data
   */
  std::unique_ptr<NNTrainer::TensorsQueue> train_tensors_queue,
    valid_tensors_queue;
  /**
   * @brief Nntrainer dataset
   */
  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_valid;

  ml::train::RunStats train_stats;
  ml::train::RunStats valid_stats;

private:
  bool stop_model_training{FALSE};
  std::string model_config;
  std::string model_save_path; /**< Model is finally stored */
  std::string model_load_path; /**< Path to load an exisiting model to use for
                                  training a new model */
  unsigned int num_pushed_samples{0U}; /**< The number of samples pushed by
                                 NNStreamer(tensor_trainer) */
  unsigned int num_epochs;             /**< The number of epoch */
  unsigned int total_num_samples;      /**< Total number of samples received for
                                          creating model */
  unsigned int num_inputs; /**< The number of tensors used as input in the
                              received a sample */
  unsigned int num_labels; /**< The number of tensors used as label in the
                              received a sample */
  unsigned int num_training_samples;   /**< The number of training samples to be
                                          taken for training model */
  unsigned int num_validation_samples; /**< The number of validation samples to
                                          be taken for validation model */
  unsigned int tensors_inputsize[NNS_TENSOR_SIZE_LIMIT];
  unsigned int
    num_tensors; /**< The number of tensors in the received a sample */
  GstTensorTrainerEventNotifier *notifier{
    nullptr}; /**< a handle of event notify */
  std::unique_ptr<ml::train::Model> model;
};

/**
 * @brief Stores input tensors to the queue
 */
class TensorsQueue {
public:
  /**
   * @brief Construct a new TensorsQueue object
   * @param _num_of_samples Total number of samples received for creating
   * model
   * @param _num_of_inputs The number of tensors used as input in the received a
   * sample
   * @param _num_of_labels The number of tensors used as label in the received a
   * sample
   * @param _tensors_size[] size of each tensor in a sample
   */
  TensorsQueue(unsigned int _num_of_samples, unsigned int _num_of_inputs,
               unsigned int _num_of_labels, unsigned int _tensors_size[]);

  /**
   * @brief Destroy the TensorsQueue object
   */
  ~TensorsQueue();

  /**
   * @brief Get push count
   */
  unsigned int getPushCount() { return push_count; }

  /**
   * @brief Initialize push count
   */
  void initPushCount() { push_count = 0; }

  /**
   * @brief Pop tensors from Queue, nntrainer calls when it needs data.
   *
   * @param input input dataint
   * @param label label data
   * @param last set TRUE if data is last
   */
  void pop(float **input, float **label, bool *last);

  /**
   * @brief Push input tensor to Queue
   *
   * @param input input tensors from sub-plugin
   */
  void push(const GstTensorMemory *input);

  /**
   * @brief Check if queue is empty
   */
  bool isQueueEmpty();

  /**
   * @brief Check if queue is full
   */
  bool isQueueFull();

private:
  unsigned int queue_size{0U};
  unsigned int queue_front{0U};
  unsigned int queue_rear{0U};
  unsigned int push_count{0U}; /**< The number of samples pushed to queue by
                              NNStreamer(tensor_trainer) */
  unsigned int pop_count{0U};  /**< The number of pop from the queue for pushing
                              samples to nntrainer */
  unsigned int
    input_size[NNS_TENSOR_SIZE_LIMIT]; /**< feature size * data type */
  unsigned int label_size[NNS_TENSOR_SIZE_LIMIT];
  unsigned int num_of_samples; /**< Total number of samples received for
                                     training model */
  unsigned int
    num_of_inputs; /**< The number of tensors in the received a sample */
  unsigned int num_of_labels; /**< The number of tensors used as label in the
                              received a sample */

  std::vector<TensorsData> queue; /**< Manage multiple inputs and labels data */
  std::mutex queue_lock;
  std::condition_variable data_full;
  std::condition_variable data_empty;
};
} // namespace NNTrainer
