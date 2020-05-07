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

#include "neuralnet.h"
#include "nntrainer_internal.h"
#include "nntrainer_log.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Function to create Network::NeuralNetwork object.
 */
static int nn_object(ml_nnmodel_h *model) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel = new ml_nnmodel;
  nnmodel->magic = ML_NNTRAINER_MAGIC;

  *model = nnmodel;

  try {
    nnmodel->network = new nntrainer::NeuralNetwork();
  } catch (const char *e) {
    ml_loge("Error: heap exception: %s", e);
    status = ML_ERROR_CANNOT_ASSIGN_ADDRESS;
    delete nnmodel;
  }

  return status;
}

int ml_nnmodel_construct(ml_nnmodel_h *model) {
  int status = ML_ERROR_NONE;

  status = nn_object(model);
  return status;
}

int ml_nnmodel_construct_with_conf(const char *model_conf,
                                   ml_nnmodel_h *model) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;

  std::ifstream conf_file(model_conf);
  if (!conf_file.good()) {
    ml_loge("Error: Cannot open model configuration file : %s", model_conf);
    return ML_ERROR_INVALID_PARAMETER;
  }

  status = ml_nnmodel_construct(model);

  nnmodel = (ml_nnmodel *)(*model);

  nntrainer::NeuralNetwork *nn = (nnmodel)->network;

  nn->setConfig(model_conf);
  return status;
}

int ml_nnmodel_compile(ml_nnmodel_h model) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  nntrainer::NeuralNetwork *NN;
  NN = nnmodel->network;
  status = NN->checkValidation();
  if (status != ML_ERROR_NONE)
    return status;
  status = NN->init();
  return status;
}

int ml_nnmodel_train(ml_nnmodel_h model) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);
  nntrainer::NeuralNetwork *NN;
  NN = nnmodel->network;
  status = NN->train();
  return status;
}

int ml_nnmodel_destruct(ml_nnmodel_h model) {
  int status = ML_ERROR_NONE;
  ml_nnmodel *nnmodel;

  ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model);

  nntrainer::NeuralNetwork *NN;
  NN = nnmodel->network;
  NN->finalize();
  delete NN;

  return status;
}

int ml_nnlayer_create(ml_nnlayer_h *layer, ml_layer_type_e type) {
  int status = ML_ERROR_NONE;
  ml_nnlayer *nnlayer = new ml_nnlayer;
  nnlayer->magic = ML_NNTRAINER_MAGIC;
  *layer = nnlayer;
  try {
    switch (type) {
    case ML_LAYER_TYPE_INPUT:
      nnlayer->layer = new nntrainer::InputLayer();
      nnlayer->layer->setType(nntrainer::LAYER_IN);
      break;
    case ML_LAYER_TYPE_FC:
      nnlayer->layer = new nntrainer::FullyConnectedLayer();
      nnlayer->layer->setType(nntrainer::LAYER_FC);
      break;
    default:
      delete nnlayer;
      ml_loge("Error: Unknown layer type");
      status = ML_ERROR_INVALID_PARAMETER;
      break;
    }
  } catch (const char *e) {
    ml_loge("Error: heap exception: %s", e);
    status = ML_ERROR_CANNOT_ASSIGN_ADDRESS;
    delete nnlayer;
  }

  return status;
}

int ml_nnlayer_delete(ml_nnlayer_h layer) {
  int status = ML_ERROR_NONE;
  ml_nnlayer *nnlayer;

  ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer);

  nntrainer::Layer *NL;
  NL = nnlayer->layer;

  delete NL;
  delete nnlayer;

  return status;
}

int ml_nnlayer_set_property(ml_nnlayer_h layer, ml_layer_property_e key,
                            const char *value) {
  int status = ML_ERROR_NONE;
  ml_nnlayer *nnlayer;
  ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer);

  nntrainer::Layer *NL;
  NL = nnlayer->layer;

  status = NL->setProperty(key, value);

  return status;
}

#ifdef __cplusplus
}
#endif
