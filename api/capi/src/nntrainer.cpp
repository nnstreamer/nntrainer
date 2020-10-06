/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file nntrainer.cpp
 * @date 02 April 2020
 * @brief NNTrainer C-API Wrapper.
 *        This allows to construct and control NNTrainer Model.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <databuffer_factory.h>
#include <layer_factory.h>
#include <layer_internal.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_internal.h>
#include <nntrainer_log.h>
#include <optimizer_factory.h>
#include <parse_util.h>
#include <sstream>
#include <stdarg.h>
#include <string.h>

/**
 * @brief   Global lock for nntrainer C-API
 * @details This lock ensures that ml_train_model_destroy is thread safe. All
 *          other API functions use the mutex from their object handle. However
 *          for destroy, object mutex cannot be used as their handles are
 *          destroyed at destroy.
 */
std::mutex GLOCK;

/**
 * @brief   Adopt the lock to the current scope for the object
 */
#define ML_TRAIN_ADOPT_LOCK(obj, obj_lock) \
  std::lock_guard<std::mutex> obj_lock(obj->m, std::adopt_lock)

/**
 * @brief     function to wrap an exception to predefined error value
 * @param[in] func must be wrapped inside lambda []() -> int otherwise compile
 * error will be raised
 * @retval    errorno
 */
template <typename F> static int nntrainer_exception_boundary(F &&func) {
  int status = ML_ERROR_NONE;

  /**< Exception boundary for cpp exceptions */
  /// @note aware that some exception are inheritance of others so should be
  /// caught before than some
  try {
    status = func();
  } catch (nntrainer::exception::not_supported &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (std::invalid_argument &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (std::range_error &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (std::out_of_range &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (std::logic_error &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_INVALID_PARAMETER;
  } catch (std::bad_alloc &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_OUT_OF_MEMORY;
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return ML_ERROR_UNKNOWN;
  } catch (...) {
    ml_loge("unknown error type thrown");
    return ML_ERROR_UNKNOWN;
  }

  /**< Exception boundary for specialized error code */
  /// @todo deprecate this with #233
  switch (status) {
  case ML_ERROR_BAD_ADDRESS:
    return ML_ERROR_OUT_OF_MEMORY;
  case ML_ERROR_RESULT_OUT_OF_RANGE:
    return ML_ERROR_INVALID_PARAMETER;
  default:
    return status;
  }
}

typedef std::function<int()> returnable;

/**
 * @brief std::make_shared wrapped with exception boundary
 *
 * @tparam Tv value type.
 * @tparam Tp pointer type.
 * @tparam Types args used to construct
 * @param target pointer
 * @param args args
 * @return int error value. ML_ERROR_OUT_OF_MEMORY if fail
 */
template <typename Tv, typename Tp, typename... Types>
static int exception_bounded_make_shared(Tp &target, Types... args) {
  returnable f = [&]() {
    target = std::make_shared<Tv>(args...);
    return ML_ERROR_NONE;
  };

  return nntrainer_exception_boundary(f);
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function to create Network::NeuralNetwork object.
 */
static int nn_object(ml_train_model_h *model) {
  int status = ML_ERROR_NONE;

  if (model == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  ml_train_model *nnmodel = new ml_train_model;
  nnmodel->magic = ML_NNTRAINER_MAGIC;
  nnmodel->optimizer = NULL;
  nnmodel->dataset = NULL;

  *model = nnmodel;

  status =
    exception_bounded_make_shared<nntrainer::NeuralNetwork>(nnmodel->network);
  if (status != ML_ERROR_NONE) {
    ml_loge("Error: creating nn object failed");
    delete nnmodel;
  }

  return status;
}

int ml_train_model_construct(ml_train_model_h *model) {
  int status = ML_ERROR_NONE;

  check_feature_state();

  returnable f = [&]() { return nn_object(model); };

  status = nntrainer_exception_boundary(f);
  return status;
}

int ml_train_model_construct_with_conf(const char *model_conf,
                                       ml_train_model_h *model) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  returnable f;

  status = ml_train_model_construct(model);
  if (status != ML_ERROR_NONE)
    return status;

  std::ifstream conf_file(model_conf);
  if (!conf_file.good()) {
    ml_train_model_destroy(*model);
    ml_loge("Error: Cannot open model configuration file : %s", model_conf);
    return ML_ERROR_INVALID_PARAMETER;
  }

  nnmodel = (ml_train_model *)(*model);
  NN = nnmodel->network;

  f = [&]() { return NN->loadFromConfig(model_conf); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    ml_train_model_destroy(*model);
  }

  return status;
}

int ml_train_model_compile(ml_train_model_h model, ...) {
  int status = ML_ERROR_NONE;
  const char *data;
  ml_train_model *nnmodel;
  returnable f;
  std::shared_ptr<nntrainer::NeuralNetwork> NN;

  check_feature_state();

  ML_TRAIN_VERIFY_VALID_HANDLE(model);

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, model);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }
  va_end(arguments);

  {
    ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
    ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
    NN = nnmodel->network;
  }

  f = [&]() { return NN->setProperty(arg_list); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  f = [&]() { return NN->init(); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  return status;
}

int ml_train_model_run(ml_train_model_h model, ...) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  const char *data;
  std::shared_ptr<nntrainer::NeuralNetwork> NN;

  check_feature_state();

  ML_TRAIN_VERIFY_VALID_HANDLE(model);

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, model);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  {
    ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
    ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
    NN = nnmodel->network;
  }

  returnable f = [&]() { return NN->train(arg_list); };
  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_model_destroy(ml_train_model_h model) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;

  check_feature_state();

  {
    ML_TRAIN_GET_VALID_MODEL_LOCKED_RESET(nnmodel, model);
    ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
  }

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  NN = nnmodel->network;

  if (nnmodel->optimizer) {
    ML_TRAIN_RESET_VALIDATED_HANDLE(nnmodel->optimizer);
    delete nnmodel->optimizer;
  }

  if (nnmodel->dataset) {
    ML_TRAIN_RESET_VALIDATED_HANDLE(nnmodel->dataset);
    delete nnmodel->dataset;
  }

  for (auto &x : nnmodel->layers_map) {
    ML_TRAIN_RESET_VALIDATED_HANDLE(x.second);
    delete (x.second);
  }
  nnmodel->layers_map.clear();

  delete nnmodel;

  return status;
}

static int ml_train_model_get_summary_util(ml_train_model_h model,
                                           ml_train_summary_type_e verbosity,
                                           std::stringstream &ss) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  std::shared_ptr<nntrainer::NeuralNetwork> NN;

  {
    ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
    ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);

    NN = nnmodel->network;
  }

  returnable f = [&]() {
    NN->printPreset(ss, verbosity);
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);
  return status;
}

int ml_train_model_get_summary(ml_train_model_h model,
                               ml_train_summary_type_e verbosity,
                               char **summary) {
  int status = ML_ERROR_NONE;
  std::stringstream ss;

  check_feature_state();

  if (summary == nullptr) {
    ml_loge("summary pointer is null");
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = ml_train_model_get_summary_util(model, verbosity, ss);
  if (status != ML_ERROR_NONE) {
    ml_loge("failed make a summary: %d", status);
    return status;
  }

  std::string str = ss.str();
  const std::string::size_type size = str.size();

  if (size == 0) {
    ml_logw("summary is empty for the model!");
  }

  *summary = (char *)malloc((size + 1) * sizeof(char));
  memcpy(*summary, str.c_str(), size + 1);

  return status;
}

int ml_train_model_add_layer(ml_train_model_h model, ml_train_layer_h layer) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  ml_train_layer *nnlayer;

  check_feature_state();

  ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
  ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
  ML_TRAIN_GET_VALID_LAYER_LOCKED(nnlayer, layer);
  ML_TRAIN_ADOPT_LOCK(nnlayer, layer_lock);

  if (nnlayer->in_use) {
    ml_loge("Layer already in use.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  std::shared_ptr<nntrainer::Layer> NL;

  NN = nnmodel->network;
  NL = nnlayer->layer;

  if (nnmodel->layers_map.count(NL->getName())) {
    ml_loge("It is not allowed to add layer with same name: %s",
            NL->getName().c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  returnable f = [&]() { return NN->addLayer(NL); };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  nnmodel->layers_map.insert({NL->getName(), nnlayer});
  nnlayer->in_use = true;
  return status;
}

int ml_train_model_set_optimizer(ml_train_model_h model,
                                 ml_train_optimizer_h optimizer) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  ml_train_optimizer *nnopt;

  check_feature_state();

  ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
  ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
  ML_TRAIN_GET_VALID_OPT_LOCKED(nnopt, optimizer);
  ML_TRAIN_ADOPT_LOCK(nnopt, opt_lock);

  if (nnopt->in_use) {
    ml_loge("Optimizer already in use.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  std::shared_ptr<nntrainer::Optimizer> opt;

  NN = nnmodel->network;
  opt = nnopt->optimizer;

  returnable f = [&]() { return NN->setOptimizer(opt); };

  status = nntrainer_exception_boundary(f);
  if (status == ML_ERROR_NONE) {
    nnopt->in_use = true;
    if (nnmodel->optimizer)
      nnmodel->optimizer->in_use = false;
    nnmodel->optimizer = nnopt;
  }

  return status;
}

int ml_train_model_set_dataset(ml_train_model_h model,
                               ml_train_dataset_h dataset) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  ml_train_dataset *nndataset;

  check_feature_state();

  ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
  ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
  ML_TRAIN_GET_VALID_DATASET_LOCKED(nndataset, dataset);
  ML_TRAIN_ADOPT_LOCK(nndataset, dataset_lock);

  if (nndataset->in_use) {
    ml_loge("Dataset already in use.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  std::shared_ptr<nntrainer::DataBuffer> data;

  NN = nnmodel->network;
  data = nndataset->data_buffer;

  returnable f = [&]() { return NN->setDataBuffer(data); };

  status = nntrainer_exception_boundary(f);
  if (status == ML_ERROR_NONE) {
    nndataset->in_use = true;
    if (nnmodel->dataset)
      nnmodel->dataset->in_use = false;
    nnmodel->dataset = nndataset;
  }

  return status;
}

int ml_train_model_get_layer(ml_train_model_h model, const char *layer_name,
                             ml_train_layer_h *layer) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;

  check_feature_state();

  ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
  ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);

  std::unordered_map<std::string, ml_train_layer *>::iterator layer_iter =
    nnmodel->layers_map.find(std::string(layer_name));
  /** if layer found in layers_map, return layer */
  if (layer_iter != nnmodel->layers_map.end()) {
    *layer = layer_iter->second;
    return status;
  }

  /**
   * if layer not found in layers_map, get layer from NeuralNetwork,
   * wrap it in struct nnlayer, add new entry in layer_map and then return
   */
  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  std::shared_ptr<nntrainer::Layer> NL;

  NN = nnmodel->network;
  returnable f = [&]() { return NN->getLayer(layer_name, &NL); };
  status = nntrainer_exception_boundary(f);

  if (status != ML_ERROR_NONE)
    return status;

  ml_train_layer *nnlayer = new ml_train_layer;
  nnlayer->magic = ML_NNTRAINER_MAGIC;
  nnlayer->layer = NL;
  nnlayer->in_use = true;
  nnmodel->layers_map.insert({NL->getName(), nnlayer});

  *layer = nnlayer;
  return status;
}

int ml_train_layer_create(ml_train_layer_h *layer, ml_train_layer_type_e type) {
  int status = ML_ERROR_NONE;
  ml_train_layer *nnlayer;

  check_feature_state();

  nnlayer = new ml_train_layer;
  nnlayer->magic = ML_NNTRAINER_MAGIC;
  nnlayer->in_use = false;

  returnable f = [&]() {
    nnlayer->layer = nntrainer::createLayer(ml_layer_to_nntrainer_type(type));
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    delete nnlayer;
    ml_loge("Error: Create layer failed");
  } else {
    *layer = nnlayer;
  }

  return status;
}

int ml_train_layer_destroy(ml_train_layer_h layer) {
  int status = ML_ERROR_NONE;
  ml_train_layer *nnlayer;

  check_feature_state();

  {
    ML_TRAIN_GET_VALID_LAYER_LOCKED_RESET(nnlayer, layer);
    ML_TRAIN_ADOPT_LOCK(nnlayer, layer_lock);

    if (nnlayer->in_use) {
      ml_loge("Cannot delete layer already added in a model."
              "Delete model will delete this layer.");
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

  delete nnlayer;

  return status;
}

int ml_train_layer_set_property(ml_train_layer_h layer, ...) {
  int status = ML_ERROR_NONE;
  ml_train_layer *nnlayer;
  const char *data;
  std::shared_ptr<nntrainer::Layer> NL;

  check_feature_state();

  ML_TRAIN_VERIFY_VALID_HANDLE(layer);

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, layer);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  {
    ML_TRAIN_GET_VALID_LAYER_LOCKED(nnlayer, layer);
    ML_TRAIN_ADOPT_LOCK(nnlayer, layer_lock);

    NL = nnlayer->layer;
  }

  returnable f = [&]() { return NL->setProperty(arg_list); };
  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_optimizer_create(ml_train_optimizer_h *optimizer,
                              ml_train_optimizer_type_e type) {
  int status = ML_ERROR_NONE;

  check_feature_state();

  ml_train_optimizer *nnopt = new ml_train_optimizer;
  nnopt->magic = ML_NNTRAINER_MAGIC;
  nnopt->in_use = false;

  returnable f = [&]() {
    nnopt->optimizer =
      nntrainer::createOptimizer(ml_optimizer_to_nntrainer_type(type));
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    delete nnopt;
    ml_loge("creating optimizer failed");
  } else {
    *optimizer = nnopt;
  }

  return status;
}

int ml_train_optimizer_destroy(ml_train_optimizer_h optimizer) {
  int status = ML_ERROR_NONE;
  ml_train_optimizer *nnopt;

  check_feature_state();

  {
    ML_TRAIN_GET_VALID_OPT_LOCKED_RESET(nnopt, optimizer);
    ML_TRAIN_ADOPT_LOCK(nnopt, optimizer_lock);

    if (nnopt->in_use) {
      ml_loge("Cannot delete optimizer already set to a model."
              "Delete model will delete this optimizer.");
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

  delete nnopt;
  return status;
}

int ml_train_optimizer_set_property(ml_train_optimizer_h optimizer, ...) {
  int status = ML_ERROR_NONE;
  ml_train_optimizer *nnopt;
  const char *data;
  std::shared_ptr<nntrainer::Optimizer> opt;

  check_feature_state();

  ML_TRAIN_VERIFY_VALID_HANDLE(optimizer);

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, optimizer);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  {
    ML_TRAIN_GET_VALID_OPT_LOCKED(nnopt, optimizer);
    ML_TRAIN_ADOPT_LOCK(nnopt, optimizer_lock);

    opt = nnopt->optimizer;
  }

  returnable f = [&]() { return opt->setProperty(arg_list); };

  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_dataset_create_with_generator(ml_train_dataset_h *dataset,
                                           ml_train_datagen_cb train_cb,
                                           ml_train_datagen_cb valid_cb,
                                           ml_train_datagen_cb test_cb) {
  int status = ML_ERROR_NONE;

  check_feature_state();

  if (!train_cb)
    return ML_ERROR_INVALID_PARAMETER;

  std::shared_ptr<nntrainer::DataBuffer> data_buffer;

  returnable f = [&]() {
    data_buffer =
      nntrainer::createDataBuffer(nntrainer::DataBufferType::GENERATOR);
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    ml_loge("Error: Create dataset failed");
    return status;
  }

  f = [&]() {
    return data_buffer->setFunc(nntrainer::BufferType::BUF_TRAIN, train_cb);
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  f = [&]() {
    return data_buffer->setFunc(nntrainer::BufferType::BUF_VAL, valid_cb);
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  f = [&]() {
    return data_buffer->setFunc(nntrainer::BufferType::BUF_TEST, test_cb);
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  ml_train_dataset *nndataset = new ml_train_dataset;
  nndataset->magic = ML_NNTRAINER_MAGIC;
  nndataset->data_buffer = data_buffer;
  nndataset->in_use = false;

  *dataset = nndataset;
  return status;
}

int ml_train_dataset_create_with_file(ml_train_dataset_h *dataset,
                                      const char *train_file,
                                      const char *valid_file,
                                      const char *test_file) {
  int status = ML_ERROR_NONE;

  check_feature_state();

  std::shared_ptr<nntrainer::DataBuffer> data_buffer;
  std::shared_ptr<nntrainer::DataBufferFromDataFile> data_buffer_file;

  returnable f = [&]() {
    data_buffer = nntrainer::createDataBuffer(nntrainer::DataBufferType::FILE);
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    ml_loge("Error: Create dataset failed");
    return status;
  }

  data_buffer_file =
    std::static_pointer_cast<nntrainer::DataBufferFromDataFile>(data_buffer);

  if (train_file) {
    status = data_buffer_file->setDataFile(train_file, nntrainer::DATA_TRAIN);
    if (status != ML_ERROR_NONE) {
      return status;
    }
  } else {
    ml_loge("Train data file must be valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (valid_file) {
    status = data_buffer_file->setDataFile(valid_file, nntrainer::DATA_VAL);
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  if (test_file) {
    status = data_buffer_file->setDataFile(test_file, nntrainer::DATA_TEST);
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  ml_train_dataset *nndataset = new ml_train_dataset;
  nndataset->magic = ML_NNTRAINER_MAGIC;
  nndataset->data_buffer = data_buffer;
  nndataset->in_use = false;

  *dataset = nndataset;
  return status;
}

int ml_train_dataset_set_property(ml_train_dataset_h dataset, ...) {
  int status = ML_ERROR_NONE;
  ml_train_dataset *nndataset;
  void *data;
  std::shared_ptr<nntrainer::DataBuffer> data_buffer;

  check_feature_state();

  ML_TRAIN_VERIFY_VALID_HANDLE(dataset);

  std::vector<void *> arg_list;
  va_list arguments;
  va_start(arguments, dataset);

  while ((data = va_arg(arguments, void *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  {
    ML_TRAIN_GET_VALID_DATASET_LOCKED(nndataset, dataset);
    ML_TRAIN_ADOPT_LOCK(nndataset, dataset_lock);

    data_buffer = nndataset->data_buffer;
  }

  returnable f = [&]() { return data_buffer->setProperty(arg_list); };
  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_dataset_destroy(ml_train_dataset_h dataset) {
  int status = ML_ERROR_NONE;
  ml_train_dataset *nndataset;

  check_feature_state();

  {
    ML_TRAIN_GET_VALID_DATASET_LOCKED_RESET(nndataset, dataset);
    ML_TRAIN_ADOPT_LOCK(nndataset, dataset_lock);

    if (nndataset->in_use) {
      ml_loge("Cannot delete dataset already set to a model."
              "Delete model will delete this dataset.");
      return ML_ERROR_INVALID_PARAMETER;
    }
  }

  delete nndataset;
  return status;
}

#ifdef __cplusplus
}
#endif
