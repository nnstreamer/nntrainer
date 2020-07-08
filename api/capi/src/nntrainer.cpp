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
#include <ml-api-common.h>
#include <neuralnet.h>
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
  try {
    return func();
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
}

#ifdef __cplusplus
extern "C" {
#endif

typedef std::function<int()> returnable;

/**
 * @brief Function to create Network::NeuralNetwork object.
 */
static int nn_object(ml_nnmodel_h *model) {
  int status = ML_ERROR_NONE;

  if (model == NULL)
    return ML_ERROR_INVALID_PARAMETER;

  ml_nnmodel *nnmodel = new ml_nnmodel;
  nnmodel->magic = ML_NNTRAINER_MAGIC;
  nnmodel->optimizer = NULL;

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

int ml_nnmodel_construct(ml_nnmodel_h *model) {
  int status = ML_ERROR_NONE;
  returnable f = [&]() { return nn_object(model); };

  status = nntrainer_exception_boundary(f);
  return status;
}

int ml_nnmodel_construct_with_conf(const char *model_conf,
                                   ml_nnmodel_h *model) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;
  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  returnable f;

  std::ifstream conf_file(model_conf);
  if (!conf_file.good()) {
    ml_loge("Error: Cannot open model configuration file : %s", model_conf);
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = ml_nnmodel_construct(model);
  if (status != ML_ERROR_NONE)
    return status;

  nnmodel = (ml_nnmodel *)(*model);
  NN = nnmodel->network;

  f = [&]() { return NN->setConfig(model_conf); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    ml_nnmodel_destruct(*model);
    return status;
  }

  f = [&]() { return NN->loadFromConfig(); };
  status = nntrainer_exception_boundary(f);
  if (status != ML_ERROR_NONE) {
    ml_nnmodel_destruct(*model);
  }

  return status;
}

int ml_nnmodel_compile(ml_nnmodel_h model, ...) {
  int status = ML_ERROR_NONE;
  const char *data;
  ml_nnmodel *nnmodel;
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

int ml_nnmodel_train_with_file(ml_nnmodel_h model, ...) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;
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

int ml_nnmodel_train_with_generator(ml_nnmodel_h model,
                                    bool (*train_func)(float *, float *, int *),
                                    bool (*val_func)(float *, float *, int *),
                                    bool (*test_func)(float *, float *, int *),
                                    ...) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;
  const char *data;

  std::vector<std::string> arg_list;
  va_list arguments;
  va_start(arguments, (test_func));
  while ((data = va_arg(arguments, const char *))) {
    arg_list.push_back(data);
  }

  va_end(arguments);

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  NN = nnmodel->network;

  returnable f = [&]() {
    return NN->train((train_func), (val_func), (test_func), arg_list);
  };

  status = nntrainer_exception_boundary(f);

  return status;
}

int ml_nnmodel_destruct(ml_nnmodel_h model) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;

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
  for (auto iter = nnmodel->layers.begin(); iter != nnmodel->layers.end();
       ++iter)
    delete (*iter);
  delete nnmodel;

  return status;
}

int ml_nnmodel_add_layer(ml_nnmodel_h model, ml_nnlayer_h layer) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;
  ml_nnlayer *nnlayer;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer);

  std::shared_ptr<nntrainer::NeuralNetwork> NN;
  std::shared_ptr<nntrainer::Layer> NL;

  NN = nnmodel->network;
  NL = nnlayer->layer;

  returnable f = [&]() { return NN->addLayer(NL); };

  status = nntrainer_exception_boundary(f);
  if (status == ML_ERROR_NONE) {
    nnlayer->in_use = true;
    nnmodel->layers.push_back(nnlayer);
  }

  return status;
}

int ml_nnmodel_set_optimizer(ml_nnmodel_h model, ml_nnopt_h optimizer) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;
  ml_nnopt *nnopt;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  ML_NNTRAINER_CHECK_OPT_VALIDATION(nnopt, optimizer);

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

int ml_nnlayer_create(ml_nnlayer_h *layer, ml_layer_type_e type) {
  int status = ML_ERROR_NONE;
  returnable f;
  ml_nnlayer *nnlayer = new ml_nnlayer;
  nnlayer->magic = ML_NNTRAINER_MAGIC;
  *layer = nnlayer;

  try {
    switch (type) {
    case ML_LAYER_TYPE_INPUT:
      nnlayer->layer = std::make_shared<nntrainer::InputLayer>();
      break;
    case ML_LAYER_TYPE_FC:
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
  return status;
}

int ml_nnlayer_delete(ml_nnlayer_h layer) {
  int status = ML_ERROR_NONE;
  ml_nnlayer *nnlayer;

  ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer);

  if (nnlayer->in_use) {
    ml_loge("Cannot delete layer already added in a model."
            "Delete model will delete this layer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  delete nnlayer;

  return status;
}

int ml_nnlayer_set_property(ml_nnlayer_h layer, ...) {
  int status = ML_ERROR_NONE;
  ml_nnlayer *nnlayer;
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

int ml_nnoptimizer_create(ml_nnopt_h *optimizer, const char *type) {
  int status = ML_ERROR_NONE;

  ml_nnopt *nnopt = new ml_nnopt;
  nnopt->magic = ML_NNTRAINER_MAGIC;
  nnopt->optimizer = std::make_shared<nntrainer::Optimizer>();
  nnopt->in_use = false;

  *optimizer = nnopt;

  returnable f = [&]() {
    return nnopt->optimizer->setType(
      (nntrainer::OptType)parseType(type, nntrainer::TOKEN_OPT));
  };
  status = nntrainer_exception_boundary(f);

  if (status != ML_ERROR_NONE) {
    delete nnopt;
  }

  return status;
}

int ml_nnoptimizer_delete(ml_nnopt_h optimizer) {
  int status = ML_ERROR_NONE;
  ml_nnopt *nnopt;

  ML_NNTRAINER_CHECK_OPT_VALIDATION(nnopt, optimizer);

  if (nnopt->in_use) {
    ml_loge("Cannot delete optimizer already set to a model."
            "Delete model will delete this optimizer.");
    return ML_ERROR_INVALID_PARAMETER;
  }

  delete nnopt;
  return status;
}

int ml_nnoptimizer_set_property(ml_nnopt_h optimizer, ...) {
  int status = ML_ERROR_NONE;
  ml_nnopt *nnopt;
  const char *data;
  nnopt = (ml_nnopt *)optimizer;
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

#ifdef __cplusplus
}
#endif
