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
#include <pthread.h>
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

static int nntrainer_model_invoke(const GstTensorTrainerFramework *fw,
                                  const GstTensorTrainerProperties *prop,
                                  void *private_data,
                                  const GstTensorMemory *input) {
  NNTrainer::InputTensorsInfo *data = nullptr;
  NNTrainer::NNTrainerTrain *nntrainer =
    reinterpret_cast<NNTrainer::NNTrainerTrain *>(private_data);
  UNUSED(fw);
  ml_logd("<called>");

  if (!nntrainer) {
    ml_loge("Failed get nntrainer");
    return -1;
  }
  UNUSED(prop);

  if (nntrainer->train_data->push_count == nntrainer->num_train_samples &&
      nntrainer->valid_data->push_count == nntrainer->num_valid_samples) {
    ml_logd("data is full");
    return 0;
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

  NNTrainer::TensorData tensor_data;
  int64_t idx = 0, i = 0;
  char *p_data = nullptr;
  for (i = 0; i < data->num_inputs; i++) {
    ml_logd("input[%d]:%p, size:%zd\n", i, input[i].data, input[i].size);
    p_data = new char[input[idx].size];
    std::memcpy(p_data, input[idx].data, input[idx].size);
    tensor_data.inputs.emplace_back(p_data);
    ml_logd("input[%d].data = p %p\n", idx, (input[idx].data));
    ml_logd("tensor_data.inputs[%d] = %p\n", idx, tensor_data.inputs[idx]);
    idx++;
  }
  for (i = 0; i < data->num_labels; i++) {
    p_data = new char[input[idx].size];
    std::memcpy(p_data, input[idx].data, input[idx].size);
    tensor_data.labels.emplace_back(p_data);
    idx++;
  }

  data->tensor_data.emplace_back(tensor_data);
  data->push_count++;

  ml_logd("(pop/push: %d/%d)", data->pop_count, data->push_count);

#if 0
  for (auto data : data->tensor_data) {
    for (auto inputs : data.inputs) {
      ml_logd("##I addr:%p", inputs);
    }
    for (auto labels : data.labels) {
      ml_logd("##L addr:%p", labels);
    }
  }
#endif

  if (data->is_mutex_locked && data->push_count > data->pop_count) {
    pthread_mutex_lock(&data->mutex);
    ml_logd("send signal");
    pthread_cond_signal(&data->cond);
    pthread_mutex_unlock(&data->mutex);
  }

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

  if (data->push_count <= data->pop_count) {
    pthread_mutex_lock(&data->mutex);
    data->is_mutex_locked = TRUE;
    ml_logd("locked, need to wait for more data");
    pthread_cond_wait(&data->cond, &data->mutex);
    ml_logd("unlocked, get data");
    pthread_mutex_unlock(&data->mutex);
    data->is_mutex_locked = FALSE;
  }

  ml_logd("num_inputs: %d, num_labels: %d", data->num_inputs, data->num_labels);

  int64_t i = 0;
  int idx = data->pop_count;
  ml_logd("pop idx: %d", idx);

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

  ml_logd("(pop/push: %d/%d)", data->pop_count, data->push_count);

  if (data->pop_count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->pop_count = 0;

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(data->tensor_data.begin(), data->tensor_data.end(), g);
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
  is_mutex_locked(0),
  push_count(0),
  pop_count(0),
  num_samples(_num_samples),
  num_inputs(_num_inputs),
  num_labels(_num_labels) {

  ml_logd("<called>");

  tensor_data.reserve(_num_samples);
  pthread_mutex_init(&mutex, NULL);
  pthread_cond_init(&cond, NULL);

  int64_t idx = 0, i = 0;
  for (i = 0; i < num_inputs; i++) {
    input_size[i] = _tensors_inputsize[idx++];
    ml_logd("input_size[%d]=%d", i, input_size[i]);
  }
  for (i = 0; i < num_labels; i++) {
    label_size[i] = _tensors_inputsize[idx++];
    ml_logd("label_size[%d]=%d", i, label_size[i]);
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
  train_complete_cond = prop->train_complete_cond;

  ml_logd("num_inputs: %d", num_inputs);
  ml_logd("num_labels: %d", num_labels);
  ml_logd("num_train_samples: %d", num_train_samples);
  ml_logd("num_valid_samples: %d", num_valid_samples);
  ml_logd("model_config: %s", model_config.c_str());
  ml_logd("model_config: %s", model_save_path.c_str());
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
  ml_logd("<called>");
  UNUSED(fw);
  UNUSED(prop);
  UNUSED(private_data);

  fw_info->name = subplugin_name;
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
