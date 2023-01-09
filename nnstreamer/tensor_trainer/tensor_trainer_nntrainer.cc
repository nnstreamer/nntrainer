/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer tensor_trainer subplugin for nntrainer
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file   tensor_trainer_nntrainer.cc
 * @date   02 Dec 2022
 * @brief  NNStreamer tensor_trainer subplugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Hyunil Park <hyunil46.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include "tensor_trainer_nntrainer.hh"
#include <cstring>
#include <iostream>
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
 * @brief invoke function
 * tensor_trainer call this function to send tensor_data.
 * For epoch, (number of trin samples + number of valid samples) * epoch
 * data should be received.
 * Sub-plugin don't keep dataset for epoch.
 */
static int nntrainer_model_invoke(const GstTensorTrainerFramework *fw,
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

  nntrainer->num_invoke++;
  ml_logd("Received data (%d/%d(total))", nntrainer->num_invoke,
          nntrainer->total_num_samples);
  if (nntrainer->total_num_samples < nntrainer->num_invoke) {
    ml_logd("Already received all data required for the train, "
            "but invoke is called");
    return 0;
  }

  if (nntrainer->train_data->push_count == nntrainer->num_train_samples &&
      nntrainer->valid_data->push_count == nntrainer->num_valid_samples) {
    nntrainer->train_data->push_count = nntrainer->valid_data->push_count = 0;
  }

  if (nntrainer->train_data->push_count < nntrainer->num_train_samples) {
    data = nntrainer->train_data.get();
    ml_logd("#### T-Data ####");
  } else if (nntrainer->valid_data->push_count < nntrainer->num_valid_samples) {
    data = nntrainer->valid_data.get();
    ml_logd("#### V-Data ####");
  }

  ml_logd("number of inputs(%d) and labels(%d)", nntrainer->num_inputs,
          nntrainer->num_labels);

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

  ml_logd("(pop/push: %d/%d)", data->pop_count, data->push_count);
  ml_logd("T-pushed:%d/%d, V-pushed:%d/%d\n", nntrainer->train_data->push_count,
          nntrainer->num_train_samples, nntrainer->valid_data->push_count,
          nntrainer->num_valid_samples);

  ml_logd("<leaved>");
  return 0;
}

int getSample(float **input, float **label, bool *last, void *user_data) {

  auto data = reinterpret_cast<NNTrainer::InputTensorsInfo *>(user_data);

  ml_logd("<called>");
  ml_logd("(pop/push: %d/%d)", data->pop_count, data->push_count);
  pid_t pid = getpid();
  pid_t tid = syscall(SYS_gettid);

  ml_logd("<called>");
  ml_logd("pid[%d], tid[%d]", pid, tid);

  ml_logd("front:%d, rear:%d", data->queue_front, data->queue_rear);

  ml_logd("num_inputs: %d, num_labels: %d", data->num_inputs, data->num_labels);

  unsigned int i = 0;
  unsigned int idx = data->queue_front;
  ml_logd("Delete, queue_front: %d", idx);

  for (i = 0; i < data->num_inputs; i++) {
    ml_logd("memcpy Addr %p, %p, size=%d\n", *(input + i),
            data->tensor_data[idx].inputs[i], data->input_size[i]);
    std::memcpy(*(input + i), data->tensor_data[idx].inputs[i],
                data->input_size[i]);
  }
  for (i = 0; i < data->num_labels; i++) {
    ml_logd("memcpy Addr %p, %p, size=%d", *(label + i),
            data->tensor_data[idx].labels[i], data->label_size[i]);
    std::memcpy(*(label + i), data->tensor_data[idx].labels[i],
                data->label_size[i]);
  }

  data->pop_count++;
  data->queue_count--;
  data->queue_front = (data->queue_front + 1) % data->queue_size;

  ml_logd("(pop/push: %d/%d)", data->pop_count, data->push_count);

  if (data->pop_count < data->num_samples) { // train or valid num samples
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
    num_train_samples, num_inputs, num_labels, tensors_inputsize);
  valid_data = std::make_unique<NNTrainer::InputTensorsInfo>(
    num_valid_samples, num_inputs, num_labels, tensors_inputsize);

  if (num_train_samples) {
    dataset_train = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             getSample, train_data.get());
  }
  if (num_valid_samples) {
    dataset_valid = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             getSample, valid_data.get());
  }
  ml_logd("<leave>");
}

NNTrainer::InputTensorsInfo::InputTensorsInfo(int64_t _num_samples,
                                              int64_t _num_inputs,
                                              int64_t _num_labels,
                                              int64_t _tensors_inputsize[]) :
  is_data_wait_locked(0),
  is_data_full_locked(0),
  queue_front(0),
  queue_rear(0),
  queue_count(0),
  push_count(0),
  pop_count(0),
  num_samples(_num_samples),
  num_inputs(_num_inputs),
  num_labels(_num_labels) {

  ml_logd("<called>");
  const int min_queue_size = 30;
  queue_size = (_num_samples > min_queue_size) ? min_queue_size : _num_samples;
  ml_logd("queue_size:%d", queue_size);
  tensor_data.reserve(queue_size);

  int64_t idx = 0, i = 0;
  for (i = 0; i < num_inputs; i++) {
    input_size[i] = _tensors_inputsize[idx++];
    ml_logd("input_size[%d]=%d", i, input_size[i]);
  }
  for (i = 0; i < num_labels; i++) {
    label_size[i] = _tensors_inputsize[idx++];
    ml_logd("label_size[%d]=%d", i, label_size[i]);
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
  ml_logd("num_tensors: %d", num_tensors);

  for (i = 0; i < num_tensors; i++) {
    tensors_inputsize[i] = gst_tensor_info_get_size(&prop->input_meta.info[i]);
    ml_logd("tensors_inputsize[%d]:%d", i, tensors_inputsize[i]);
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
  num_train_samples = prop->num_train_samples;
  num_valid_samples = prop->num_valid_samples;
  model_save_path = prop->model_save_path;
  num_epochs = prop->num_epochs;
  train_complete_cond = prop->train_complete_cond;
  is_train_complete = FALSE;
  total_num_samples = (num_train_samples + num_valid_samples) * num_epochs;

  ml_logd("num_inputs: %d", num_inputs);
  ml_logd("num_labels: %d", num_labels);
  ml_logd("num_train_samples: %d", num_train_samples);
  ml_logd("num_valid_samples: %d", num_valid_samples);
  ml_logd("num_epochs: %d", num_epochs);
  ml_logd("Total number of data to be received: %d", total_num_samples);
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

static int nntrainer_model_train(const GstTensorTrainerFramework *fw,
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
    is_train_complete = TRUE;
    ml_logd("send train_complete_cond signal");
    g_cond_signal(train_complete_cond);

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
  num_invoke(0), model_config(_model_config) {
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
    fw_info->train_complete = nntrainer->is_train_complete;
  ml_logd("<leave>");
  return 0;
}

static GstTensorTrainerFramework NNS_Trainer_support_nntrainer = {
  .version = GST_TENSOR_TRAINER_FRAMEWORK_V1,
  .create = nntrainer_model_construct,
  .destroy = nntrainer_model_destructor,
  .train = nntrainer_model_train,
  .invoke = nntrainer_model_invoke,
  .getFrameworkInfo = nntrainer_getFrameworkInfo};

void init_subplugin_nntrainer(void) {
  nnstreamer_trainer_probe(&NNS_Trainer_support_nntrainer);
}

void fini_subplugin_nntrainer(void) {
  nnstreamer_trainer_exit(&NNS_Trainer_support_nntrainer);
}
