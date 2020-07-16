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
 * @date 13 April 2020
 * @brief NNTrainer C-API Internal Header.
 *        This allows to construct and control NNTrainer Model.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 *
 * @note This API is not stable and still experimental.
 */

#ifndef __NNTRAINER_INTERNAL_H__
#define __NNTRAINER_INTERNAL_H__

#include <layer.h>
#include <neuralnet.h>
#include <nntrainer.h>
#include <nntrainer_log.h>
#include <optimizer.h>
#include <string>
#include <unordered_map>

#define ML_NNTRAINER_MAGIC 0x777F888F

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Struct to wrap neural network layer for the API
 */
typedef struct {
  uint magic;
  std::shared_ptr<nntrainer::Layer> layer;
  bool in_use;
} ml_train_layer;

/**
 * @brief Struct to wrap neural network optimizer for the API
 */
typedef struct {
  uint magic;
  std::shared_ptr<nntrainer::Optimizer> optimizer;
  bool in_use;
} ml_train_optimizer;

/**
 * @brief Struct to wrap data buffer for the API
 */
typedef struct {
  uint magic;
  std::shared_ptr<nntrainer::DataBuffer> data_buffer;
  bool in_use;
} ml_train_dataset;

/**
 * @brief Struct to wrap neural network model for the API
 */
typedef struct {
  uint magic;
  std::shared_ptr<nntrainer::NeuralNetwork> network;
  std::unordered_map<std::string, ml_train_layer *> layers_map;
  ml_train_optimizer *optimizer;
  ml_train_dataset *dataset;
} ml_train_model;

/**
 * @brief     Check validity of the user passed arguments
 */
#define ML_NNTRAINER_CHECK_VALIDATION(obj, obj_h, obj_type, obj_name) \
  do {                                                                \
    if (!obj_h) {                                                     \
      ml_loge("Error: Invalid Parameter : %s is empty.", obj_name);   \
      return ML_ERROR_INVALID_PARAMETER;                              \
    }                                                                 \
    obj = (obj_type *)obj_h;                                          \
    if (obj->magic != ML_NNTRAINER_MAGIC) {                           \
      ml_loge("Error: Invalid Parameter : %s is invalid.", obj_name); \
      return ML_ERROR_INVALID_PARAMETER;                              \
    }                                                                 \
  } while (0)

#define ML_NNTRAINER_CHECK_MODEL_VALIDATION(nnmodel, model) \
  ML_NNTRAINER_CHECK_VALIDATION(nnmodel, model, ml_train_model, "model")

#define ML_NNTRAINER_CHECK_LAYER_VALIDATION(nnlayer, layer) \
  ML_NNTRAINER_CHECK_VALIDATION(nnlayer, layer, ml_train_layer, "layer")

#define ML_NNTRAINER_CHECK_OPT_VALIDATION(nnopt, opt) \
  ML_NNTRAINER_CHECK_VALIDATION(nnopt, opt, ml_train_optimizer, "optimizer")

#define ML_NNTRAINER_CHECK_DATASET_VALIDATION(nndataset, dataset) \
  ML_NNTRAINER_CHECK_VALIDATION(nndataset, dataset, ml_train_dataset, "dataset")

/**
 * @brief Get neural network layer from the model with the given name.
 * @details Use this function to get already created Neural Network Layer. The
 * returned layer must not be deleted as it is owned by the model.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] layer_name Name of the already created layer.
 * @param[out] layer The NNTrainer Layer handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 * @retval #ML_ERROR_CANNOT_ASSIGN_ADDRESS Cannot assign object.
 */
int ml_train_model_get_layer(ml_train_model_h model, const char *layer_name,
                             ml_train_layer_h *layer);
/**
 * @brief Get all neural network layer names from the model.
 * @details Use this function to get already created Neural Network Layer names.
 * This can be used to obtain layers when model is defined with ini file.
 * @note The caller must free the list of the layer names.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[out] layers_name List of names of layers in the model ended with NULL.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 * @retval #ML_ERROR_CANNOT_ASSIGN_ADDRESS Cannot assign object.
 */
int ml_train_model_get_all_layer_names(ml_train_model_h model,
                                       const char **layers_name[]);

/**
 * @brief Callback function to notify completion of training of the model.
 * @param[in] model The NNTrainer model handler.
 * @param[in] data Internal data to be given to the callback, cb.
 */
typedef void (*ml_train_run_cb)(ml_train_model_h model, void *data);

/**
 * @brief Train the neural network model asynchronously.
 * @details Use this function to train the compiler neural network model with
 * the passed training hyperparameters. The callback will be called once the
 * requested training, validation and testing is completed.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler.
 * @param[in] cb The callback handler to be called after training finishes.
 * @param[in] data Internal data to be given to the callback, cb.
 * @param[in] ...  Hyperparmeter for train model
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_model_run_async(ml_train_model_h model, ml_train_run_cb cb,
                             void *data, ...);

/**
 * @brief Insert layer at the specific location of the existing layers in neural
 * network model.
 * @details Use this function to insert a layer to the model.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] layer The NNTrainer layer handler
 * @param[in] input_layer_names List of layers ended with NULL, which will
 * provide input to the layer being inserted.
 * @param[in] output_layer_names List of layers ended with NULL, which will
 * receive input to the layer being inserted.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 * @details If length of @a input_layer_names is more than 1, the layer to be
 * inserted should support multiple inputs. Otherwise
 * #ML_ERROR_INVALID_PARAMETER is returned. If the layer in @a
 * output_layer_names already have input connection, then they should support
 * multiple inputs. Otherwise #ML_ERROR_INVALID_PARAMETER is returned. If length
 * of @a output_layer_names is 0, then this layer will be treated as one of the
 * output layers, and a loss will be attached to this based on network
 * configuration. If both @a input_layer_names and @a output_layer_names are
 * empty, then this layer is attached at the end of the output layer of the
 * network. In case of multiple output layers, this layer is attached next to
 * the last created output layer.
 */
int ml_train_model_insert_layer(ml_train_model_h model, ml_train_layer_h layer,
                                const char *input_layer_names[],
                                const char *output_layer_names[]);

#ifdef __cplusplus
}
#endif /* __cplusplus */

/**
 * @brief Convert nntrainer API optimizer type to neural network optimizer type
 * @param[in] type Optimizer type API enum
 * @return nntrainer::OptType optimizer type
 */
nntrainer::OptType
ml_optimizer_to_nntrainer_type(ml_train_optimizer_type_e type);

#endif
