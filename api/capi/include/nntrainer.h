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
 * @file nntrainer.h
 * @date 02 April 2020
 * @brief NNTrainer C-API Header.
 *        This allows to construct and control NNTrainer Model.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNTRAINER_H__
#define __TIZEN_MACHINELEARNING_NNTRAINER_H__

#include <stdbool.h>
#include <stddef.h>

#include <ml-api-common.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNTRAINER_MODULE
 * @{
 */

/**
 * @brief A handle of an NNTrainer model.
 * @since_tizen 6.x
 */
typedef void *ml_train_model_h;

/**
 * @brief A handle of an NNTrainer layer.
 * @since_tizen 6.x
 */
typedef void *ml_train_layer_h;

/**
 * @brief A handle of an NNTrainer optimizer.
 * @since_tizen 6.x
 */
typedef void *ml_train_optimizer_h;

/**
 * @brief Enumeration for the neural network layer type of NNTrainer.
 * @since_tizen 6.x
 */
typedef enum {
  ML_TRAIN_LAYER_TYPE_INPUT = 0,    /**< Input Layer */
  ML_TRAIN_LAYER_TYPE_FC,           /**< Fully Connected Layer */
  ML_TRAIN_LAYER_TYPE_UNKNOWN = 999 /**< Unknown Layer */
} ml_train_layer_type_e;

/**
 * @brief Enumeration for the neural network optimizer type of NNTrainer.
 * @since_tizen 6.x
 */
typedef enum {
  ML_TRAIN_OPTIMIZER_TYPE_ADAM = 0, /**< Adam Optimizer */
  ML_TRAIN_OPTIMIZER_TYPE_SGD, /**< Stochastic Gradient Descent Optimizer */
  ML_TRAIN_OPTIMIZER_TYPE_UNKNOWN = 999 /**< Unknown Optimizer */
} ml_train_optimizer_type_e;

/**
 * @brief Enumeration for the neural network summary verbosity of NNTrainer
 * @since_tizen 6.x
 */
typedef enum {
  ML_TRAIN_SUMMARY_MODEL = 0, /**< Overview of model
                                   summary with one-line layer information*/
  ML_TRAIN_SUMMARY_LAYER, /**< Detailed model summary with layer properties */
  ML_TRAIN_SUMMARY_TENSOR /**< Model summary layer's including weight
                             information */
} ml_train_summary_type_e;

/**
 * @brief Constructs the neural network model.
 * @details Use this function to create Neural Netowrk Model.
 * @since_tizen 6.x
 * @param[out] model The NNTrainer Model handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_CANNOT_ASSIGN_ADDRESS Cannot assign object.
 */
int ml_train_model_construct(ml_train_model_h *model);

/**
 * @brief Construct the neural network model with the given configuration file.
 * @details Use this function to create neural network model with the given
 * configuration file.
 * @since_tizen 6.x
 * @param[in] model_conf The location of nntrainer model configuration file.
 * @param[out] model The NNTrainer model handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_model_construct_with_conf(const char *model_conf,
                                       ml_train_model_h *model);

/**
 * @brief initialize the neural network model.
 * @details Use this function to initialize neural network model. Once compiled,
 * addition of new layers is not permitted. Further, updating the properties of
 * added layers is restricted.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] type The NNTrainer loss type.
 * @param[in] ... hyper parmeter for compile model
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_model_compile(ml_train_model_h model, ...);

/**
 * @brief train the neural network model.
 * @details Use this function to train neural network model
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] ...  hyper parmeter for train model
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_nnmodel_train_with_file(ml_train_model_h model, ...);

/**
 * @brief train the neural network model.
 * @details Use this function to train neural network model
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] train_func function pointer for train
 * @param[in] val_func function pointer for val
 * @param[in] test_func function pointer for test
 * @param[in] ...  hyper parmeter for train model
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_nnmodel_train_with_generator(ml_train_model_h model,
                                    bool (*train_func)(float *, float *, int *),
                                    bool (*val_func)(float *, float *, int *),
                                    bool (*test_func)(float *, float *, int *),
                                    ...);

/**
 * @brief Destructs the neural network model.
 * @details Use this function to destroy Neural Network Model.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_model_destroy(ml_train_model_h model);

/**
 * @brief Get the summary of the neural network model.
 * @details Use this function to get the summary of the neural network model.
 * @since_tizen 6.x
 * @remarks If the function succeeds, @a summary should be released using
 * free().
 * @param[in] model The NNTrainer model handler to get summary.
 * @param[in] verbosity Verbose level of the summary
 * @param[out] summary The summary of the current model. Avoid logic to parse
 * and exploit @a summary if possible.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_nnmodel_get_summary(ml_train_model_h model,
                                 ml_train_summary_type_e verbosity,
                                 char **summary);

/**
 * @brief Add layer at the last of the existing layers in neural network model.
 * @details Use this function to add a layer to the model. This transfers the
 * ownership of the layer to the network. No need to destroy the layer if it
 * belongs to a model.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] layer The NNTrainer layer handler
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_model_add_layer(ml_train_model_h model, ml_train_layer_h layer);

/**
 * @brief Set the neural network optimizer.
 * @details Use this function to set Neural Network Optimizer. Unsets the
 * previous optimizer if any. This transfers the ownership of the optimizer to
 * the network. No need to destroy the optimizer if it is to a model.
 * @since_tizen 6.x
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] optimizer The NNTrainer Optimizer handler
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_model_set_optimizer(ml_train_model_h model,
                                 ml_train_optimizer_h optimizer);

/**
 * @brief Create the neural network layer.
 * @details Use this function to create Neural Netowrk Layer.
 * @since_tizen 6.x
 * @param[out] layer The NNTrainer Layer handler from the given description.
 * @param[in]  type The NNTrainer Layer type
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 * @retval #ML_ERROR_CANNOT_ASSIGN_ADDRESS Cannot assign object.
 */
int ml_train_layer_create(ml_train_layer_h *layer, ml_train_layer_type_e type);

/**
 * @brief destroy the neural network layer.
 * @details Use this function to destroy Neural Network Layer. Fails if layer is
 * owned by a model.
 * @since_tizen 6.x
 * @param[in] layer The NNTrainer layer handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_layer_destroy(ml_train_layer_h layer);

/**
 * @brief Set the neural network layer Property.
 * @details Use this function to set Neural Netowrk Layer Property.
 * @since_tizen 6.x
 * @param[in] layer The NNTrainer Layer handler from the given description.
 * @param[in]  ... Property values with NULL for termination.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_layer_set_property(ml_train_layer_h layer, ...);

/**
 * @brief Create the neural network optimizer.
 * @details Use this function to create Neural Netowrk Optimizer.
 * @since_tizen 6.x
 * @param[out] optimizer The NNTrainer Optimizer handler
 * @param[in] type The NNTrainer Optimizer type
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_optimizer_create(ml_train_optimizer_h *optimizer,
                              ml_train_optimizer_type_e type);

/**
 * @brief destroy the neural network optimizer.
 * @details Use this function to destroy Neural Netowrk Optimizer. Fails if
 * optimizer is owned by a model.
 * @since_tizen 6.x
 * @param[in] optimizer The NNTrainer optimizer handler from the given
 * description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter.
 */
int ml_train_optimizer_destroy(ml_train_optimizer_h optimizer);

/**
 * @brief Set the neural network optimizer property.
 * @details Use this function to set Neural Netowrk Optimizer Property.
 * @since_tizen 6.x
 * @param[in] optimizer The NNTrainer Optimizer handler from the given
 * description.
 * @param[in]  ... Property values with NULL at the end.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_optimizer_set_property(ml_train_optimizer_h optimizer, ...);

/**
 * @}
 */
#ifdef __cplusplus
}

#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNTRAINER_H__ */
