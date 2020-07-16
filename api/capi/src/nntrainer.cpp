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
 * @bug No known bugs except for NYI items
 */
#include <databuffer.h>
#include <databuffer_file.h>
#include <databuffer_func.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_internal.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <stdarg.h>
#include <string.h>

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
  } catch (nntrainer::exception::invalid_property &e) {
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
  case ML_ERROR_CANNOT_ASSIGN_ADDRESS:
  case ML_ERROR_BAD_ADDRESS:
    return ML_ERROR_OUT_OF_MEMORY;
  case ML_ERROR_RESULT_OUT_OF_RANGE:
    return ML_ERROR_INVALID_PARAMETER;
  default:
    return status;
  }
}

#ifdef __cplusplus
extern "C" {
#endif

typedef std::function<int()> returnable;

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

  try {
    nnmodel->network = std::make_shared<nntrainer::NeuralNetwork>();
  } catch (std::bad_alloc &e) {
    ml_loge("Error: heap exception: %s", e.what());
    status = ML_ERROR_OUT_OF_MEMORY;
    delete nnmodel;
  }

  return status;
}

int ml_train_model_construct(ml_train_model_h *model) {
  int status = ML_ERROR_NONE;
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

  std::ifstream conf_file(model_conf);
  if (!conf_file.good()) {
    ml_loge("Error: Cannot open model configuration file : %s", model_conf);
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = ml_train_model_construct(model);
  if (status != ML_ERROR_NONE)
    return status;

  nnmodel = (ml_train_model *)(*model);
  NN = nnmodel->network;

  f = [&]() { return NN->setConfig(model_conf); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    ml_train_model_destroy(*model);
    return status;
  }

  f = [&]() { return NN->loadFromConfig(); };
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

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, model);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }
  va_end(arguments);

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  NN = nnmodel->network;

  f = [&]() { return NN->setProperty(arg_list); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  f = [&]() { return NN->init(); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE)
    return status;

  f = [&]() { return NN->checkValidation(); };
  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_model_run(ml_train_model_h model, ...) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  const char *data;

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, model);
  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  NN = nnmodel->network;

  returnable f = [&]() { return NN->train(arg_list); };

  status = nntrainer_exception_boundary(f);
  return status;
}

int ml_train_model_destroy(ml_train_model_h model) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  NN = nnmodel->network;

  returnable f = [&]() {
    NN->finalize();
    return ML_ERROR_NONE;
  };

  status = nntrainer_exception_boundary(f);

  if (nnmodel->optimizer)
    delete nnmodel->optimizer;
  if (nnmodel->dataset)
    delete nnmodel->dataset;
  for (auto &x : nnmodel->layers_map)
    delete (x.second);
  nnmodel->layers_map.clear();
  delete nnmodel;

  return status;
}

int ml_train_model_add_layer(ml_train_model_h model, ml_train_layer_h layer) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  ml_train_layer *nnlayer;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer);

  if (nnlayer->in_use) {
    ml_loge("Layer already in use.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  std::shared_ptr<nntrainer::Layer> NL;

  NN = nnmodel->network;
  NL = nnlayer->layer;

  returnable f = [&]() { return NN->addLayer(NL); };

  status = nntrainer_exception_boundary(f);
  if (status == ML_ERROR_NONE) {
    nnlayer->in_use = true;
    nnmodel->layers_map.insert({NL->getName(), nnlayer});
  }

  return status;
}

int ml_train_model_set_optimizer(ml_train_model_h model,
                                 ml_train_optimizer_h optimizer) {
  int status = ML_ERROR_NONE;
  ml_train_model *nnmodel;
  ml_train_optimizer *nnopt;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  ML_NNTRAINER_CHECK_OPT_VALIDATION(nnopt, optimizer);

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

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  ML_NNTRAINER_CHECK_DATASET_VALIDATION(nndataset, dataset);

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
  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  std::shared_ptr<nntrainer::Layer> NL;

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
  returnable f;
  ml_train_layer *nnlayer = new ml_train_layer;
  nnlayer->magic = ML_NNTRAINER_MAGIC;

  try {
    switch (type) {
    case ML_TRAIN_LAYER_TYPE_INPUT:
      nnlayer->layer = std::make_shared<nntrainer::InputLayer>();
      break;
    case ML_TRAIN_LAYER_TYPE_FC:
      nnlayer->layer = std::make_shared<nntrainer::FullyConnectedLayer>();
      break;
    default:
      delete nnlayer;
      ml_loge("Error: Unknown layer type");
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  } catch (std::bad_alloc &e) {
    ml_loge("Error: heap exception: %s", e.what());
    status = ML_ERROR_OUT_OF_MEMORY;
    delete nnlayer;
  }

  nnlayer->in_use = false;
  *layer = nnlayer;
  return status;
}

int ml_train_layer_destroy(ml_train_layer_h layer) {
  int status = ML_ERROR_NONE;
  ml_train_layer *nnlayer;

  ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer);

  if (nnlayer->in_use) {
    ml_loge("Cannot delete layer already added in a model."
            "Delete model will delete this layer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  delete nnlayer;

  return status;
}

int ml_train_layer_set_property(ml_train_layer_h layer, ...) {
  int status = ML_ERROR_NONE;
  ml_train_layer *nnlayer;
  const char *data;

  ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer);

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, layer);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  std::shared_ptr<nntrainer::Layer> NL;
  NL = nnlayer->layer;

  returnable f = [&]() { return NL->setProperty(arg_list); };
  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_optimizer_create(ml_train_optimizer_h *optimizer,
                              ml_train_optimizer_type_e type) {
  int status = ML_ERROR_NONE;

  ml_train_optimizer *nnopt = new ml_train_optimizer;
  nnopt->magic = ML_NNTRAINER_MAGIC;
  nnopt->optimizer = std::make_shared<nntrainer::Optimizer>();
  nnopt->in_use = false;

  *optimizer = nnopt;

  returnable f = [&]() {
    return nnopt->optimizer->setType(ml_optimizer_to_nntrainer_type(type));
  };
  status = nntrainer_exception_boundary(f);

  if (status != ML_ERROR_NONE) {
    delete nnopt;
  }

  return status;
}

int ml_train_optimizer_destroy(ml_train_optimizer_h optimizer) {
  int status = ML_ERROR_NONE;
  ml_train_optimizer *nnopt;

  ML_NNTRAINER_CHECK_OPT_VALIDATION(nnopt, optimizer);

  if (nnopt->in_use) {
    ml_loge("Cannot delete optimizer already set to a model."
            "Delete model will delete this optimizer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  delete nnopt;
  return status;
}

int ml_train_optimizer_set_property(ml_train_optimizer_h optimizer, ...) {
  int status = ML_ERROR_NONE;
  ml_train_optimizer *nnopt;
  const char *data;
  nnopt = (ml_train_optimizer *)optimizer;
  ML_NNTRAINER_CHECK_OPT_VALIDATION(nnopt, optimizer);

  std::vector<std::string> arg_list;

  va_list arguments;
  va_start(arguments, optimizer);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  std::shared_ptr<nntrainer::Optimizer> opt;
  opt = nnopt->optimizer;

  returnable f = [&]() { return opt->setProperty(arg_list); };

  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_dataset_create_with_generator(ml_train_dataset_h *dataset,
                                           ml_train_datagen_cb train_cb,
                                           ml_train_datagen_cb valid_cb,
                                           ml_train_datagen_cb test_cb) {
  int status = ML_ERROR_NONE;

  std::shared_ptr<nntrainer::DataBufferFromCallback> data_buffer =
    std::make_shared<nntrainer::DataBufferFromCallback>();

  status = data_buffer->setFunc(nntrainer::BUF_TRAIN, train_cb);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = data_buffer->setFunc(nntrainer::BUF_VAL, valid_cb);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = data_buffer->setFunc(nntrainer::BUF_TEST, test_cb);
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

  std::shared_ptr<nntrainer::DataBufferFromDataFile> data_buffer =
    std::make_shared<nntrainer::DataBufferFromDataFile>();

  if (train_file) {
    status = data_buffer->setDataFile(train_file, nntrainer::DATA_TRAIN);
    if (status != ML_ERROR_NONE) {
      return status;
    }
  } else {
    ml_loge("Train data file must be valid.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (valid_file) {
    status = data_buffer->setDataFile(valid_file, nntrainer::DATA_VAL);
    if (status != ML_ERROR_NONE) {
      return status;
    }
  }

  if (test_file) {
    status = data_buffer->setDataFile(test_file, nntrainer::DATA_TEST);
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
  const char *data;

  ML_NNTRAINER_CHECK_DATASET_VALIDATION(nndataset, dataset);

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, dataset);

  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  returnable f = [&]() {
    return nndataset->data_buffer->setProperty(arg_list);
  };
  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_train_dataset_destroy(ml_train_dataset_h dataset) {
  int status = ML_ERROR_NONE;
  ml_train_dataset *nndataset;

  ML_NNTRAINER_CHECK_DATASET_VALIDATION(nndataset, dataset);

  if (nndataset->in_use) {
    ml_loge("Cannot delete dataset already set to a model."
            "Delete model will delete this dataset.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  delete nndataset;
  return status;
}

#ifdef __cplusplus
}
#endif
