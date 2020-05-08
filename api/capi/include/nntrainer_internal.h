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
 * @file nntrainer_internal.h
 * @date 02 April 2020
 * @brief NNTrainer C-API Internal Header.
 *        This allows to construct and control NNTrainer Model.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNTRAINER_INTERNAL_H__
#define __NNTRAINER_INTERNAL_H__

#include <nntrainer.h>

#define ML_NNTRAINER_MAGIC 0x777F888F

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

typedef struct {
  uint magic;
  std::shared_ptr<nntrainer::NeuralNetwork> network;
} ml_nnmodel;

typedef struct {
  uint magic;
  std::shared_ptr<nntrainer::Layer> layer;
} ml_nnlayer;

#define ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model)      \
  do {                                                           \
    if (!model) {                                                \
      ml_loge("Error: Invalid Parameter : model is empty.");     \
      return ML_ERROR_INVALID_PARAMETER;                         \
    }                                                            \
    nnmodel = (ml_nnmodel *)model;                               \
    if (nnmodel->magic != ML_NNTRAINER_MAGIC) {                  \
      ml_loge("Error: Invalid Parameter : nnmodel is invalid."); \
      return ML_ERROR_INVALID_PARAMETER;                         \
    }                                                            \
  } while (0)

#define ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer)      \
  do {                                                           \
    if (!layer) {                                                \
      ml_loge("Error: Invalid Parameter : layer is empty.");     \
      return ML_ERROR_INVALID_PARAMETER;                         \
    }                                                            \
    nnlayer = (ml_nnlayer *)layer;                               \
    if (nnlayer->magic != ML_NNTRAINER_MAGIC) {                  \
      ml_loge("Error: Invalid Parameter : nnlayer is invalid."); \
      return ML_ERROR_INVALID_PARAMETER;                         \
    }                                                            \
  } while (0)

#ifdef __cplusplus
}

#endif /* __cplusplus */
#endif
