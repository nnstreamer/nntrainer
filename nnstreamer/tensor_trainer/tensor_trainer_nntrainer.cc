/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer tensor_trainer subplugin for nntrainer
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file   tensor_trainer_nntrainer.cc
 * @date   02 Dec 2022
 * @brief  NNStreamer tensor_trainer subplugin
 * @see    http://github.com/nnstreamer/nntrainer
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Hyunil Park <hyunil46.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

/**
 * # Action description and constraints
 * tensor_trainer_nntrainer.cc is a nnstreamer sub-plugine be used by
 * nnstreamer(tensor_trainer) for training model.
 *
 * ## Notice
 * 1. Models are limited to creating with configuration files.
 * 2. sub-plugin receives only pre-processed tensor data.
 * 3. The current feature behavior has been tested with MNIST.
 *
 * ## Example launch line is as below
 * gst-launch-1.0 datareposrc location=mnist_trainingSet.dat \
 * start-sample-index=3 stop-sample-index=202 epochs=5 ! \
 * other/tensors, format=static, num_tensors=2, framerate=0/1, \
 * dimensions=1:1:784:1.1:1:10:1, types=float32.float32 ! \
 * tensor_trainer framework=nntrainer model-config=mnist.ini \
 * model-save-path=model.bin input-dim=1:1:784:1,1:1:10:1 \
 * input-type=float32,float32 num-inputs=1 num-labels=1 \
 * num-training-samples=100 num-validation-samples=100 epochs=5 \
 * ! tensor_sink
 *
 * ## Notice the tensor_trainer's properties.
 * 1. model-config: model configuration file path. models are limited to
 *   creating with configuration files.
 * 2. model-save-path: model save path by query in MLOps
 * 3. input-dims:input dimensions, in case of MNIST, 1:1:784:1 is a input and
 *   1:1:10:1 is a label.
 * 4. input_type: data type. Sample(mnist_trainingSet.dat)'s data type is
 *   float32.
 * 5. num-inputs: sub-plugin supports multiple inputs, in case of MNIST,
 *   num-inputs is 1.
 * 6. num-labels: sub-plugin supports multiple labels, in case of MNIST,
 *   num-labels is 1.
 * 7. num-training-samples: Number of training samples, A sample can consist of
 *   multiple inputs and labels in tensors(in case of MNIST, all is 1), set how
 *   many samples are taken for training model.
 * 8. num-validation-samples: num-validation-samples, A sample can consist of
 *   multiple inputs and labels in tensors(in case of MNIST, all is 1), set how
 *   many samples are taken for validation model.
 * 9. epochs: epochs are repetitions of training samples and validation
 *   smaples. number of samples received for model training is
 *   (num-training-samples + num-validation-samples) * epochs
 *
 * ## Action
 * When a subplugin is loaded at runtime,
 * NNTrainerTrain's num_inputs, num_labels, num_training_samples,
 * num_validation_samples, model_save_path and num_epochs is set by
 * getNNStreamerProperties() and each tensor size is set.
 * For MNIST test, mnist_trainingSet.dat is used,
 * it consist of 1000 samples and it has 1 inputs, 1 labels, 784 and 10 size of
 * float32.
 * now, sub-plugin get num_tensors:2, tensor_size[0]:3136,
 * tensor_size[1]:40, num_inputs:1, num_labels:1, num_training_samples:100,
 * num_validation_samples:100, num_epochs:5 from nnstreamer(tensor_trainer).
 * Total number of sample to be received in sub-plugin is 1000, (100 + 100) * 5.
 *
 * nnstreamer(tensor_trainer) push a sample by call nntrainer_model_push_data()
 * and sub-plugin copy a sample to queue with incrementing
 * InputTensorsInfo::push_count. 100 samples for training, 100 samples for
 * validation comes in order.
 *
 * getSample() callback is called when nntrainer needs sample for training.
 * sub-plugin copy a sample from queue to '**inputs' and '**labels'
 * with incrementing InputTensorsInfo::pop_count.
 * if InputTensorsInfo::pop_count is same with total_num_samples then
 * '*last' is set true, and 1 epochs is finished.
 *
 */

#include "tensor_trainer_nntrainer.hh"
#include <cstring>
#include <inttypes.h>
#include <iostream>
#include <limits.h>
#include <nntrainer_log.h>
#include <random>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

/**
 * @brief startup constructor
 */
void init_subplugin_nntrainer(void) __attribute__((constructor));

/**
 * @brief startdown destructor
 */
void fini_subplugin_nntrainer(void) __attribute__((destructor));

void nntrainer_thread_func(NNTrainer::NNTrainerTrain *nntrainer) {
  nntrainer->trainModel();
}

/**
 * @brief push_data function
 * tensor_trainer call this function to push tensor data.
 * For epoch, (number of trin samples + number of valid samples) * epoch
 * data should be received.
 * Sub-plugin don't keep dataset for epoch.
 */
static int nntrainer_model_push_data(const GstTensorTrainerFramework *fw,
                                     const GstTensorTrainerProperties *prop,
                                     void *private_data,
                                     const GstTensorMemory *input) {
  NNTrainer::InputTensorsInfo *data = nullptr;
  NNTrainer::NNTrainerTrain *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerTrain *>(private_data);
  UNUSED(fw);
  UNUSED(prop);
  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);

  ml_logd("<called>");
  ml_logd("pid[%d], tid[%d]", pid, tid);

  if (!nntrainer) {
    ml_loge("Failed get nntrainer");
    return -1;
  }

  nntrainer->num_push_data++;
  ml_logd("Received data (%" PRId64 "/%" PRId64 "(total))",
          nntrainer->num_push_data, nntrainer->total_num_samples);
  if (nntrainer->total_num_samples < nntrainer->num_push_data) {
    ml_logd("Already received all data required for the train, "
            "but push_data is called");
    return 0;
  }

  if (nntrainer->train_data->push_count == nntrainer->num_training_samples &&
      nntrainer->valid_data->push_count == nntrainer->num_validation_samples) {
    nntrainer->train_data->push_count = nntrainer->valid_data->push_count = 0;
  }

  if (nntrainer->train_data->push_count < nntrainer->num_training_samples) {
    data = nntrainer->train_data.get();
    ml_logd("#### T-Data ####");
  } else if (nntrainer->valid_data->push_count <
             nntrainer->num_validation_samples) {
    data = nntrainer->valid_data.get();
    ml_logd("#### V-Data ####");
  } else {
    ml_loge("Invalid push_count");
    return -1;
  }

  ml_logd("number of inputs(%" PRId64 ") and labels(%" PRId64 ")",
          nntrainer->num_inputs, nntrainer->num_labels);

  unsigned int idx = 0, i = 0;
  i = data->queue_rear;
  ml_logd("Insert, queue_rear : %d", i);
  for (auto inputs : data->tensor_data[i].inputs) {
    ml_logd("inputs : %p", inputs);
    ml_logd("input[%d]:%p, size:%zd\n", idx, input[idx].data, input[idx].size);
    std::memcpy(inputs, input[idx].data, input[idx].size);
    idx++;
  }

  for (auto labels : data->tensor_data[i].labels) {
    ml_logd("labels : %p", labels);
    ml_logd("input[%d]:%p, size:%zd\n", idx, input[idx].data, input[idx].size);
    std::memcpy(labels, input[idx].data, input[idx].size);
    idx++;
  }

  data->push_count++;
  data->queue_count++;
  data->queue_rear = (data->queue_rear + 1) % data->queue_size;
  ml_logd("front:%d, rear:%d, filled:%d", data->queue_front, data->queue_rear,
          data->queue_count);

  if (data->is_data_wait_locked && data->queue_count > 0) {
    data->data_wait.notify_one();
    ml_logd("send signal");
  }

  if (data->queue_count == data->queue_size) {
    data->is_data_full_locked = TRUE;
    ml_logd("locked, data is full");
    std::unique_lock<std::mutex> lock(data->data_full_lock);
    data->data_full.wait(lock);
    ml_logd("unlocked, queue is empty");
    data->is_data_full_locked = FALSE;
  }

  ml_logd("(pop/push: %" PRId64 "/%" PRId64 ")", data->pop_count,
          data->push_count);
  ml_logd("T-pushed: %" PRId64 "/%" PRId64 ", V-pushed:%" PRId64 "/%" PRId64
          "\n",
          nntrainer->train_data->push_count, nntrainer->num_training_samples,
          nntrainer->valid_data->push_count, nntrainer->num_validation_samples);

  ml_logd("<leaved>");
  return 0;
}

int getSample(float **input, float **label, bool *last, void *user_data) {

  auto data = reinterpret_cast<NNTrainer::InputTensorsInfo *>(user_data);

  ml_logd("<called>");
  ml_logd("(pop/push: %" PRId64 "/%" PRId64 ")", data->pop_count,
          data->push_count);
  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);

  ml_logd("<called>");
  ml_logd("pid[%d], tid[%d]", pid, tid);
  ml_logd("front:%d, rear:%d", data->queue_front, data->queue_rear);
  ml_logd("num_inputs: %" PRId64 ", num_labels: %" PRId64 "", data->num_inputs,
          data->num_labels);

  unsigned int i = 0;
  unsigned int idx = data->queue_front;
  ml_logd("Delete, queue_front: %d", idx);

  for (i = 0; i < data->num_inputs; i++) {
    ml_logd("memcpy Addr %p, %p, size=%" PRId64 "\n", *(input + i),
            data->tensor_data[idx].inputs[i], data->input_size[i]);
    std::memcpy(*(input + i), data->tensor_data[idx].inputs[i],
                data->input_size[i]);
  }
  for (i = 0; i < data->num_labels; i++) {
    ml_logd("memcpy Addr %p, %p, size=%" PRId64 "", *(label + i),
            data->tensor_data[idx].labels[i], data->label_size[i]);
    std::memcpy(*(label + i), data->tensor_data[idx].labels[i],
                data->label_size[i]);
  }

  data->pop_count++;
  data->queue_count--;
  data->queue_front = (data->queue_front + 1) % data->queue_size;

  ml_logd("(pop/push: %" PRId64 "/%" PRId64 ")", data->pop_count,
          data->push_count);

  if (data->pop_count < data->total_num_samples) { // train or valid num samples
    *last = false;
  } else {
    *last = true;
    data->pop_count = 0;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data->tensor_data.begin(), data->tensor_data.end(), g);
  }

  if (data->is_data_full_locked && data->queue_count > 0) {
    data->data_full.notify_one();
    ml_logd("send signal");
  }
  ml_logd("front:%d, rear:%d, filled:%d", data->queue_front, data->queue_rear,
          data->queue_count);

  /* epoch is complete */
  if (data->pop_count == 0)
    return 0;

  /* to avoid dead lock, check is_data_full_locked */
  if (!data->is_data_full_locked && data->queue_count == 0) {
    ml_logd("locked, need to wait for more data");
    std::unique_lock<std::mutex> lock(data->data_wait_lock);
    data->is_data_wait_locked = TRUE;
    data->data_wait.wait(lock);
    ml_logd("unlocked, get data");
  }

  ml_logd("<leave>");

  return 0;
}

void NNTrainer::NNTrainerTrain::createDataset() {

  ml_logd("<called>");

  train_data = std::make_unique<NNTrainer::InputTensorsInfo>(
    num_training_samples, num_inputs, num_labels, tensors_inputsize);
  valid_data = std::make_unique<NNTrainer::InputTensorsInfo>(
    num_validation_samples, num_inputs, num_labels, tensors_inputsize);

  if (num_training_samples) {
    dataset_train = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             getSample, train_data.get());
  }
  if (num_validation_samples) {
    dataset_valid = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             getSample, valid_data.get());
  }
  ml_logd("<leave>");
}

NNTrainer::InputTensorsInfo::InputTensorsInfo(int64_t _total_num_samples,
                                              int64_t _num_inputs,
                                              int64_t _num_labels,
                                              int64_t _tensors_size[]) :
  is_data_wait_locked(0),
  is_data_full_locked(0),
  queue_front(0),
  queue_rear(0),
  queue_count(0),
  push_count(0),
  pop_count(0),
  total_num_samples(_total_num_samples),
  num_inputs(_num_inputs),
  num_labels(_num_labels) {

  ml_logd("<called>");
  const int min_queue_size = 30;
  queue_size =
    (_total_num_samples > min_queue_size) ? min_queue_size : _total_num_samples;
  ml_logd("queue_size:%d", queue_size);
  tensor_data.reserve(queue_size);

  int64_t idx = 0, i = 0;
  for (i = 0; i < num_inputs; i++) {
    input_size[i] = _tensors_size[idx++];
    ml_logd("input_size[%" PRId64 "]=%" PRId64 "", i, input_size[i]);
  }
  for (i = 0; i < num_labels; i++) {
    label_size[i] = _tensors_size[idx++];
    ml_logd("label_size[%" PRId64 "]=%" PRId64 "", i, label_size[i]);
  }

  unsigned int cur_queue_size = 0;

  /* make queue */
  while (cur_queue_size < queue_size) {
    NNTrainer::TensorData t_data;
    int i = 0;
    char *p_data = nullptr;
    for (i = 0; i < num_inputs; i++) {
      p_data = new char[input_size[i]];
      t_data.inputs.emplace_back(p_data);
    }
    for (i = 0; i < num_labels; i++) {
      p_data = new char[label_size[i]];
      t_data.labels.emplace_back(p_data);
    }

    tensor_data.emplace_back(t_data);
    cur_queue_size++;
  }
  ml_logd("<leave>");
}

NNTrainer::InputTensorsInfo::~InputTensorsInfo() {
  g_print("%s:%d:%s: <called>\n", __FILE__, __LINE__, __func__);

  for (auto data : tensor_data) {
    for (auto inputs : data.inputs) {
      ml_logd("free: ##I addr:%p", inputs);
      delete inputs;
    }
    for (auto labels : data.labels) {
      ml_logd("free: ##L addr:%p", labels);
      delete labels;
    }
  }
}

void NNTrainer::NNTrainerTrain::getNNStreamerProperties(
  const GstTensorTrainerProperties *prop) {

  int64_t i;
  ml_logd("<called>");

  num_tensors = prop->input_meta.num_tensors;
  ml_logd("num_tensors: %" PRId64 "", num_tensors);

  for (i = 0; i < num_tensors; i++) {
    tensors_inputsize[i] = gst_tensor_info_get_size(&prop->input_meta.info[i]);
    ml_logd("tensors_inputsize[%" PRId64 "]:%" PRId64 "", i,
            tensors_inputsize[i]);
  }
  // for mnist test
#if 0
  tensors_inputsize[1] = 40;
  // tensors_inputsize[0] = 3686400; // 3:640:480:1 float32, 3x640x480x1x4
  // tensors_inputsize[1] = 40;      // 1:1:10:1 uint8, 1x1x10x1x4
  ml_logd("for Test: tensors_inputsize[1]:%d", tensors_inputsize[1]);
#endif
  num_inputs = prop->num_inputs;
  num_labels = prop->num_labels;
  num_training_samples = prop->num_training_samples;
  num_validation_samples = prop->num_validation_samples;
  model_save_path = prop->model_save_path;
  num_epochs = prop->num_epochs;
  training_complete_cond = prop->training_complete_cond;
  is_training_complete = FALSE;
  total_num_samples =
    (num_training_samples + num_validation_samples) * num_epochs;

  ml_logd("num_inputs: %" PRId64 "", num_inputs);
  ml_logd("num_labels: %" PRId64 "", num_labels);
  ml_logd("num_training_samples: %" PRId64 "", num_training_samples);
  ml_logd("num_validation_samples: %" PRId64 "", num_validation_samples);
  ml_logd("num_epochs: %" PRId64 "", num_epochs);
  ml_logd("Total number of data to be received: %" PRId64 "",
          total_num_samples);
  ml_logd("model_config: %s", model_config.c_str());
  ml_logd("model_save_path: %s", model_save_path.c_str());
  ml_logd("<leave>");
}

static int nntrainer_model_destructor(const GstTensorTrainerFramework *fw,
                                      const GstTensorTrainerProperties *prop,
                                      void **private_data) {
  NNTrainer::NNTrainerTrain *nntrainer =
    static_cast<NNTrainer::NNTrainerTrain *>(*private_data);
  UNUSED(fw);
  ml_logd("<called>");

  if (!nntrainer)
    return -1;

  delete nntrainer;
  *private_data = NULL;
  ml_logd("<leave>");

  return 0;
}

static int
nntrainer_model_start_training(const GstTensorTrainerFramework *fw,
                               const GstTensorTrainerProperties *prop,
                               void *private_data) {
  NNTrainer::NNTrainerTrain *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerTrain *>(private_data);
  UNUSED(fw);

  ml_logd("<called>");
  if (!nntrainer) {
    ml_loge("Failed get nntrainer");
  }
  try {
    std::thread train_thread(nntrainer_thread_func, nntrainer);
    train_thread.detach();
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return -1;
  }
  ml_logd("<leave>");
  return 0;
}

void NNTrainer::NNTrainerTrain::trainModel() {
  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);

  ml_logd("<called>");
  ml_logd("pid[%d], tid[%d]", pid, tid);

  try {
    model->setProperty({"epochs=" + std::to_string(num_epochs)});
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }

  try {
    model->train();
    training_loss = model->getTrainingLoss();
    validation_loss = model->getValidationLoss();
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }
  ml_logd("training_loss: %f, validation_loss: %f", training_loss,
          validation_loss);
  try {
    ml_logd("Save_model: %s", model_save_path.c_str());
    model->save(model_save_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    is_training_complete = TRUE;
    ml_logd("send training_complete_cond signal");
    g_cond_signal(training_complete_cond);

  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }
  ml_logd("<leave>");
}

void NNTrainer::NNTrainerTrain::createModel() {
  ml_logd("<called>");
  try {
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }
  try {

    model->load(model_config,
                ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN);
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }
  try {
    model->compile();
    model->initialize();
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_valid);
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return;
  }
  ml_logd("<leave>");
}

NNTrainer::NNTrainerTrain::NNTrainerTrain(
  const GstTensorTrainerProperties *prop, const std::string &_model_config) :
  training_loss(0),
  validation_loss(0),
  num_push_data(0),
  model_config(_model_config) {
  ml_logd("<called>");
  getNNStreamerProperties(prop);
  createDataset();
  createModel();
  ml_logd("<leave>");
}

static int
nntrainer_model_construct_with_conf(const GstTensorTrainerFramework *fw,
                                    const GstTensorTrainerProperties *prop,
                                    void **private_data) {
  NNTrainer::NNTrainerTrain *nntrainer =
    static_cast<NNTrainer::NNTrainerTrain *>(*private_data);
  ml_logd("<called>");
  if (nntrainer)
    nntrainer_model_destructor(fw, prop, private_data);

  try {
    nntrainer = new NNTrainer::NNTrainerTrain(prop, prop->model_config);
  } catch (const std::exception &e) {
    ml_loge("Error %s, %s", typeid(e).name(), e.what());
    return -1;
  }

  *private_data = nntrainer;

  ml_logd("<leave>");
  return 0;
}

static int nntrainer_model_construct(const GstTensorTrainerFramework *fw,
                                     const GstTensorTrainerProperties *prop,
                                     void **private_data) {
  ml_logd("<called>");
  int status = nntrainer_model_construct_with_conf(fw, prop, private_data);

  ml_logd("<leave>");
  return status;
}

static int nntrainer_getFrameworkInfo(const GstTensorTrainerFramework *fw,
                                      const GstTensorTrainerProperties *prop,
                                      void *private_data,
                                      GstTensorTrainerFrameworkInfo *fw_info) {
  static gchar subplugin_name[] = "nntrainer";
  NNTrainer::NNTrainerTrain *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerTrain *>(private_data);
  ml_logd("<called>");
  UNUSED(fw);
  UNUSED(prop);

  fw_info->name = subplugin_name;
  if (nntrainer)
    fw_info->is_training_complete = nntrainer->is_training_complete;
  ml_logd("<leave>");
  return 0;
}

static GstTensorTrainerFramework NNS_Trainer_support_nntrainer = {
  .version = GST_TENSOR_TRAINER_FRAMEWORK_V1,
  .create = nntrainer_model_construct,
  .destroy = nntrainer_model_destructor,
  .start = nntrainer_model_start_training,
  .push_data = nntrainer_model_push_data,
  .getFrameworkInfo = nntrainer_getFrameworkInfo};

void init_subplugin_nntrainer(void) {
  nnstreamer_trainer_probe(&NNS_Trainer_support_nntrainer);
}

void fini_subplugin_nntrainer(void) {
  nnstreamer_trainer_exit(&NNS_Trainer_support_nntrainer);
}
