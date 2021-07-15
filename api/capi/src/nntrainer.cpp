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

#include <array>
#include <cstdarg>
#include <cstring>
#include <sstream>
#include <string>

#include <nntrainer.h>
#include <nntrainer_internal.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

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

/**
 * @brief Create dataset with different types of train/test/valid data source
 * @param[in] dataset dataset object to be created
 * @param[in] type type of the dataset
 * @param[in] train training data source
 * @param[in] valid validation data source
 * @param[in] test testing data source
 */
template <typename T>
static int ml_train_dataset_create(ml_train_dataset_h *dataset,
                                   ml::train::DatasetType type, T train,
                                   T valid, T test) {
  int status = ML_ERROR_NONE;

  check_feature_state();
  if (dataset == NULL) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  ml_train_dataset *nndataset = new ml_train_dataset;
  nndataset->magic = ML_NNTRAINER_MAGIC;
  nndataset->in_use = false;

  returnable f = [&]() {
    if (train != nullptr) {
      nndataset->dataset[ML_TRAIN_DATASET_MODE_TRAIN] =
        ml::train::createDataset(type, train);
    }
    if (valid != nullptr) {
      nndataset->dataset[ML_TRAIN_DATASET_MODE_VALID] =
        ml::train::createDataset(type, valid);
    }
    if (test != nullptr) {
      nndataset->dataset[ML_TRAIN_DATASET_MODE_TEST] =
        ml::train::createDataset(type, test);
    }
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    delete nndataset;
    ml_loge("Error: Create dataset failed");
  } else {
    *dataset = nndataset;
  }

  return status;
}

/**
 * @brief add ml::train::Dataset to @a dataset
 *
 * @tparam Args args needed to create the dataset
 * @param dataset dataset handle
 * @param mode target mode
 * @param type dataset type
 * @param args args needed to create the dataset
 * @retval #ML_ERROR_NONE Successful
 * @retval #ML_ERROR_INVALID_PARAMETER if parameter is invalid
 */
template <typename... Args>
static int ml_train_dataset_add_(ml_train_dataset_h dataset,
                                 ml_train_dataset_mode_e mode,
                                 ml::train::DatasetType type, Args &&... args) {
  check_feature_state();
  std::shared_ptr<ml::train::Dataset> underlying_dataset;

  returnable f = [&]() {
    underlying_dataset =
      ml::train::createDataset(type, std::forward<Args>(args)...);
    return ML_ERROR_NONE;
  };

  int status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    ml_loge("Failed to create dataset");
    return status;
  }

  if (underlying_dataset == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  ml_train_dataset *nndataset;
  ML_TRAIN_VERIFY_VALID_HANDLE(dataset);

  {
    ML_TRAIN_GET_VALID_DATASET_LOCKED(nndataset, dataset);
    ML_TRAIN_ADOPT_LOCK(nndataset, dataset_lock);

    nndataset->dataset[mode] = underlying_dataset;
  }
  return status;
}

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function to create ml::train::Model object.
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

  returnable f = [&]() {
    nnmodel->model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    delete nnmodel;
    ml_loge("Error: creating nn object failed");
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
  std::shared_ptr<ml::train::Model> m;
  returnable f;

  status = ml_train_model_construct(model);
  if (status != ML_ERROR_NONE)
    return status;

  nnmodel = (ml_train_model *)(*model);
  m = nnmodel->model;

  f = [&]() { return m->loadFromConfig(model_conf); };
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
  std::shared_ptr<ml::train::Model> m;

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
    m = nnmodel->model;
  }

  f = [&]() { return m->setProperty(arg_list); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  f = [&]() { return m->compile(); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  f = [&]() { return m->initialize(); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  return status;
}

int ml_train_model_run(ml_train_model_h model, ...) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  const char *data;
  std::shared_ptr<ml::train::Model> m;

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
    m = nnmodel->model;
  }

  returnable f = [&]() { return m->train(arg_list); };
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

  std::shared_ptr<ml::train::Model> m;
  m = nnmodel->model;

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
  std::shared_ptr<ml::train::Model> m;

  {
    ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
    ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);

    m = nnmodel->model;
  }

  returnable f = [&]() {
    m->summarize(ss, verbosity);
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
  if (*summary == nullptr) {
    ml_loge("failed to malloc");
    return ML_ERROR_OUT_OF_MEMORY;
  }
  std::memcpy(*summary, str.c_str(), size + 1);

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

  std::shared_ptr<ml::train::Model> m;
  std::shared_ptr<ml::train::Layer> l;

  m = nnmodel->model;
  l = nnlayer->layer;

  if (nnmodel->layers_map.count(l->getName())) {
    ml_loge("It is not allowed to add layer with same name: %s",
            l->getName().c_str());
    return ML_ERROR_INVALID_PARAMETER;
  }

  returnable f = [&]() { return m->addLayer(l); };

  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  nnmodel->layers_map.insert({l->getName(), nnlayer});
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

  std::shared_ptr<ml::train::Model> m;
  std::shared_ptr<ml::train::Optimizer> opt;

  m = nnmodel->model;
  opt = nnopt->optimizer;

  returnable f = [&]() { return m->setOptimizer(opt); };

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

  std::shared_ptr<ml::train::Model> m;

  m = nnmodel->model;

  returnable f = [&]() {
    auto &[train_set, valid_set, test_set] = nndataset->dataset;
    int status = ML_ERROR_NONE;
    status = m->setDataset(ml::train::DatasetModeType::MODE_TRAIN, train_set);
    if (status != ML_ERROR_NONE) {
      return status;
    }

    if (valid_set != nullptr) {
      status = m->setDataset(ml::train::DatasetModeType::MODE_VALID, valid_set);
      if (status != ML_ERROR_NONE) {
        return status;
      }
    }

    if (test_set != nullptr) {
      status = m->setDataset(ml::train::DatasetModeType::MODE_TEST, test_set);
      if (status != ML_ERROR_NONE) {
        return status;
      }
    }
    return status;
  };

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
   * if layer not found in layers_map, get layer from model,
   * wrap it in struct nnlayer, add new entry in layer_map and then return
   */
  std::shared_ptr<ml::train::Model> m;
  std::shared_ptr<ml::train::Layer> l;

  m = nnmodel->model;
  returnable f = [&]() { return m->getLayer(layer_name, &l); };
  status = nntrainer_exception_boundary(f);

  if (status != ML_ERROR_NONE)
    return status;

  ml_train_layer *nnlayer = new ml_train_layer;
  nnlayer->magic = ML_NNTRAINER_MAGIC;
  nnlayer->layer = l;
  nnlayer->in_use = true;
  nnmodel->layers_map.insert({l->getName(), nnlayer});

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
    nnlayer->layer = ml::train::createLayer((ml::train::LayerType)type);
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
  std::shared_ptr<ml::train::Layer> l;

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

    l = nnlayer->layer;
  }

  returnable f = [&]() {
    l->setProperty(arg_list);
    return ML_ERROR_NONE;
  };
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
      ml::train::createOptimizer((ml::train::OptimizerType)type);
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
  std::shared_ptr<ml::train::Optimizer> opt;

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

  returnable f = [&]() {
    opt->setProperty(arg_list);
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_dataset_create(ml_train_dataset_h *dataset) {
  return ml_train_dataset_create(dataset, ml::train::DatasetType::UNKNOWN,
                                 nullptr, nullptr, nullptr);
}

int ml_train_dataset_add_generator(ml_train_dataset_h dataset,
                                   ml_train_dataset_mode_e mode,
                                   ml_train_datagen_cb cb, void *user_data) {
  check_feature_state();
  if (cb == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ml_train_dataset_add_(dataset, mode, ml::train::DatasetType::GENERATOR,
                               cb, user_data);
}

int ml_train_dataset_add_file(ml_train_dataset_h dataset,
                              ml_train_dataset_mode_e mode, const char *file) {
  check_feature_state();
  if (file == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ml_train_dataset_add_(dataset, mode, ml::train::DatasetType::FILE,
                               file);
}

int ml_train_dataset_create_with_generator(ml_train_dataset_h *dataset,
                                           ml_train_datagen_cb train_cb,
                                           ml_train_datagen_cb valid_cb,
                                           ml_train_datagen_cb test_cb) {
  if (train_cb == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  return ml_train_dataset_create(dataset, ml::train::DatasetType::GENERATOR,
                                 train_cb, valid_cb, test_cb);
}

int ml_train_dataset_create_with_file(ml_train_dataset_h *dataset,
                                      const char *train_file,
                                      const char *valid_file,
                                      const char *test_file) {
  if (train_file == nullptr) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  return ml_train_dataset_create(dataset, ml::train::DatasetType::FILE,
                                 train_file, valid_file, test_file);
}

/**
 * @brief set property for the specific data mode, main difference from @a
 * ml_train_dataset_set_property_for_mode() is that this function returns @a
 * ML_ERROR_NOT_SUPPORTED if dataset does not exist.
 *
 * @param[in] dataset dataset
 * @param[in] mode mode
 * @param[in] args argument
 * @retval #ML_ERROR_NONE successful
 * @retval #ML_ERROR_INVALID_PARAMETER when arg is invalid
 * @retval #ML_ERROR_NOT_SUPPORTED when dataset did not exist
 */
static int
ml_train_dataset_set_property_for_mode_(ml_train_dataset_h dataset,
                                        ml_train_dataset_mode_e mode,
                                        const std::vector<void *> &args) {
  int status = ML_ERROR_NONE;
  ml_train_dataset *nndataset;

  check_feature_state();

  ML_TRAIN_VERIFY_VALID_HANDLE(dataset);

  {
    ML_TRAIN_GET_VALID_DATASET_LOCKED(nndataset, dataset);
    ML_TRAIN_ADOPT_LOCK(nndataset, dataset_lock);

    auto &db = nndataset->dataset[mode];

    returnable f = [&db, &args]() {
      int status_ = ML_ERROR_NONE;
      if (db == nullptr) {
        status_ = ML_ERROR_NOT_SUPPORTED;
        return status_;
      }

      status_ = db->setProperty(args);
      return status_;
    };

    status = nntrainer_exception_boundary(f);
  }
  return status;
}

int ml_train_dataset_set_property(ml_train_dataset_h dataset, ...) {
  std::vector<void *> arg_list;
  va_list arguments;
  va_start(arguments, dataset);

  void *data;
  while ((data = va_arg(arguments, void *))) {
    arg_list.push_back(data);
  }
  va_end(arguments);

  /// having status of ML_ERROR_NOT_SUPPORTED is not an error in this call.
  int status = ml_train_dataset_set_property_for_mode_(
    dataset, ML_TRAIN_DATASET_MODE_TRAIN, arg_list);
  if (status != ML_ERROR_NONE && status != ML_ERROR_NOT_SUPPORTED) {
    return status;
  }

  status = ml_train_dataset_set_property_for_mode_(
    dataset, ML_TRAIN_DATASET_MODE_VALID, arg_list);
  if (status != ML_ERROR_NONE && status != ML_ERROR_NOT_SUPPORTED) {
    return status;
  }

  status = ml_train_dataset_set_property_for_mode_(
    dataset, ML_TRAIN_DATASET_MODE_TEST, arg_list);
  if (status != ML_ERROR_NONE && status != ML_ERROR_NOT_SUPPORTED) {
    return status;
  }

  return ML_ERROR_NONE;
}

int ml_train_dataset_set_property_for_mode(ml_train_dataset_h dataset,
                                           ml_train_dataset_mode_e mode, ...) {
  std::vector<void *> arg_list;
  va_list arguments;
  va_start(arguments, mode);

  void *data;
  while ((data = va_arg(arguments, void *))) {
    arg_list.push_back(data);
  }
  va_end(arguments);

  int status = ml_train_dataset_set_property_for_mode_(dataset, mode, arg_list);

  return status != ML_ERROR_NONE ? ML_ERROR_INVALID_PARAMETER : ML_ERROR_NONE;
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

int ml_train_model_get_input_tensors_info(ml_train_model_h model,
                                          ml_tensors_info_h *info) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  std::shared_ptr<ml::train::Model> m;
  returnable f;

  check_feature_state();

  if (!info) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
  ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
  m = nnmodel->model;
  if (m == NULL) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::vector<ml::train::TensorDim> dims;
  f = [&]() {
    dims = m->getInputDimension();
    return ML_ERROR_NONE;
  };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = ml_tensors_info_create(info);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = ml_tensors_info_set_count(*info, dims.size());
  if (status != ML_ERROR_NONE) {
    ml_tensors_info_destroy(info);
    return status;
  }

  for (unsigned int i = 0; i < dims.size(); ++i) {
    status = ml_tensors_info_set_tensor_type(*info, i, ML_TENSOR_TYPE_FLOAT32);
    if (status != ML_ERROR_NONE) {
      ml_tensors_info_destroy(info);
      return status;
    }

    status = ml_tensors_info_set_tensor_dimension(*info, i, dims[i].getDim());
    if (status != ML_ERROR_NONE) {
      ml_tensors_info_destroy(info);
      return status;
    }
  }

  return status;
}

int ml_train_model_get_output_tensors_info(ml_train_model_h model,
                                           ml_tensors_info_h *info) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  std::shared_ptr<ml::train::Model> m;
  returnable f;

  check_feature_state();

  if (!info) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model);
  ML_TRAIN_ADOPT_LOCK(nnmodel, model_lock);
  m = nnmodel->model;
  if (m == NULL) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::vector<ml::train::TensorDim> dims;
  f = [&]() {
    dims = m->getOutputDimension();
    return ML_ERROR_NONE;
  };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = ml_tensors_info_create(info);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = ml_tensors_info_set_count(*info, dims.size());
  if (status != ML_ERROR_NONE) {
    ml_tensors_info_destroy(info);
    return status;
  }

  for (unsigned int i = 0; i < dims.size(); ++i) {
    status = ml_tensors_info_set_tensor_type(*info, i, ML_TENSOR_TYPE_FLOAT32);
    if (status != ML_ERROR_NONE) {
      ml_tensors_info_destroy(info);
      return status;
    }

    status = ml_tensors_info_set_tensor_dimension(*info, i, dims[i].getDim());
    if (status != ML_ERROR_NONE) {
      ml_tensors_info_destroy(info);
      return status;
    }
  }

  return status;
}

#ifdef __cplusplus
}
#endif
