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
 * @date 08 July 2020
 * @brief NNTrainer C-API Header.
 *        This allows to construct, control, and train a neural network model in
 * Tizen devices with nntrainer.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __TIZEN_MACHINELEARNING_NNTRAINER_H__
#define __TIZEN_MACHINELEARNING_NNTRAINER_H__

#include <stdbool.h>
#include <stddef.h>

#include <ml-api-common.h>
#include <nntrainer-api-common.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @addtogroup CAPI_ML_NNTRAINER_TRAIN_MODULE
 * @{
 */

/**
 * @brief A handle of an NNTrainer model.
 * @since_tizen 6.0
 */
typedef void *ml_train_model_h;

/**
 * @brief A handle of an NNTrainer layer.
 * @since_tizen 6.0
 */
typedef void *ml_train_layer_h;

/**
 * @brief A handle of an NNTrainer optimizer.
 * @since_tizen 6.0
 */
typedef void *ml_train_optimizer_h;

/**
 * @brief A handle of an NNTrainer dataset.
 * @since_tizen 6.0
 */
typedef void *ml_train_dataset_h;

/**
 * @brief Constructs the neural network model.
 * @details Use this function to create neural network model.
 * @since_tizen 6.0
 * @remarks If the function succeeds, @a model must be released using
 * ml_train_model_destroy().
 * @remarks %http://tizen.org/privilege/mediastorage is needed if @a model is
 * saved to media storage.
 * @remarks %http://tizen.org/privilege/externalstorage is needed if @a model is
 * saved to external storage.
 * @param[out] model The NNTrainer model handle from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_construct(ml_train_model_h *model);

/**
 * @brief Constructs the neural network model with the given configuration file.
 * @details Use this function to create neural network model with the given
 * configuration file.
 * @since_tizen 6.0
 * @remarks If the function succeeds, @a model must be released using
 * ml_train_model_destroy().
 * @param[in] model_conf The nntrainer model configuration file.
 * @param[out] model The NNTrainer model handle from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_construct_with_conf(const char *model_conf,
                                       ml_train_model_h *model);

/**
 * @brief Compiles and finalizes the neural network model with the given loss.
 * @details Use this function to initialize neural network model. Various
 * hyperparameter before compile the model can be set. Once compiled,
 * any modification to the properties of model or layers/dataset/optimizer in
 * the model will be restricted. Further, addition of layers or changing the
 * optimizer/dataset of the model will not be permitted.
 * @since_tizen 6.0
 * @param[in] model The NNTrainer model handle.
 * @param[in] ... hyperparmeters for compiling the model
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_compile(ml_train_model_h model, ...);

/**
 * @brief Trains the neural network model.
 * @details Use this function to train the compiled neural network model with
 * the passed training hyperparameters. This function will return once the
 * training, along with requested validation and testing, is completed.
 * @since_tizen 6.0
 * @param[in] model The NNTrainer model handle.
 * @param[in] ...  Hyperparmeters for train model.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_run(ml_train_model_h model, ...);

/**
 * @brief Destructs the neural network model.
 * @details Use this function to destroy neural network model.
 * @since_tizen 6.0
 * @param[in] model The NNTrainer model handle from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_destroy(ml_train_model_h model);

/**
 * @brief Gets the summary of the neural network model.
 * @details Use this function to get the summary of the neural network model.
 * @since_tizen 6.0
 * @remarks If the function succeeds, @a summary should be released using
 * free().
 * @param[in] model The NNTrainer model handle to get summary.
 * @param[in] verbosity Verbose level of the summary
 * @param[out] summary The summary of the current model. Avoid logic to parse
 * and exploit @a summary if possible.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_get_summary(ml_train_model_h model,
                               ml_train_summary_type_e verbosity,
                               char **summary);

/**
 * @brief Adds layer in neural network model.
 * @details Use this function to add a layer to the model. The layer is added to
 * the end of the existing layers in the model. This transfers the
 * ownership of the layer to the network. No need to destroy the layer once it
 * is added to a model.
 * @since_tizen 6.0
 * @param[in] model The NNTrainer model handle.
 * @param[in] layer The NNTrainer layer handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_add_layer(ml_train_model_h model, ml_train_layer_h layer);

/**
 * @brief Sets the optimizer for the neural network model.
 * @details Use this function to set neural network optimizer. This transfers
 * the ownership of the optimizer to the network. No need to destroy the
 * optimizer if it is to a model.
 * @since_tizen 6.0
 * @remarks Unsets the previously set optimizer, if any. The previously set
 * optimizer must be freed using ml_train_optimizer_destroy().
 * @param[in] model The NNTrainer model handle.
 * @param[in] optimizer The NNTrainer optimizer handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_set_optimizer(ml_train_model_h model,
                                 ml_train_optimizer_h optimizer);

/**
 * @brief Sets the dataset (data provider) for the neural network model.
 * @details Use this function to set dataset for running the model. The dataset
 * will provide training, validation and test data for the model. This transfers
 * the ownership of the dataset to the network. No need to destroy the dataset
 * once it is set to a model.
 * @since_tizen 6.0
 * @remarks Unsets the previously set dataset, if any. The previously set
 * dataset must be freed using ml_train_dataset_destroy().
 * @param[in] model The NNTrainer model handle.
 * @param[in] dataset The NNTrainer dataset handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_set_dataset(ml_train_model_h model,
                               ml_train_dataset_h dataset);

/**
 * @brief Save the model after training
 * @details Use this function to save the current model.
 * When calling with @a ML_TRAIN_MODEL_SAVE_LOAD_FLAGS_INFERENCE_PARAMS or
 * ML_TRAIN_MODEL_SAVE_LOAD_FLAGS_TRAINING_PARAMS, the model must be compiled
 * with @a ml_train_model_compile beforehand.
 *
 * @since_tizen 6.5
 * @param[in] model The NNTrainer model handle to save
 * @param[in] path_prefix  Path prefix to save the file. This function will save
 * one or number of files using the given prefix. Note that this does not state
 * the exact file name but prefix
 * @param[in] option Option flag which part of model should be saved
 * @return @c 0 on success, Otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Path is a directory without a prefix, or
 * the given path is invalid or model is not compiled.
 * @retval #ML_ERROR_NOT_SUPPORTED Given flag is not valid for the model.
 */
int ml_train_model_save(ml_train_model_h model, const char *path_prefix,
                        ml_train_model_save_load_flags_e option);

/**
 * @brief Load the model
 * @details Use this function to load the current model, model must be compiled
 * When calling with @a ML_TRAIN_MODEL_SAVE_LOAD_FLAGS_INFERENCE_PARAMS or
 * ML_TRAIN_MODEL_SAVE_LOAD_FLAGS_TRAINING_PARAMS, the model must be compiled
 * with @a ml_train_model_compile beforehand.
 *
 * @since_tizen 6.5
 * @param[in] model The NNTrainer model handle to load.
 * @param[in] path_prefix  Path prefix to load the file. Note that this does not
 * state the exact file name but prefix
 * @param[in] option Option flag which part of model should be loaded.
 * @return @c 0 on success, Otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Path is a directory without a prefix, or
 * the given path is invalid.
 * @retval #ML_ERROR_NOT_SUPPORTED Given flag is not valid for the model
 */
int ml_train_model_load(ml_train_model_h model, const char *path_prefix,
                        ml_train_model_save_load_flags_e option);

/**
 * @brief Creates a neural network layer.
 * @details Use this function to create neural network layer.
 * @since_tizen 6.0
 * @remarks If the function succeeds, @a layer must be released using
 * ml_train_layer_destroy(), if not added to a model. If added to a model, @a
 * layer is available until the model is released.
 * @param[out] layer The NNTrainer layer handle from the given description.
 * @param[in]  type The NNTrainer layer type
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_layer_create(ml_train_layer_h *layer, ml_train_layer_type_e type);

/**
 * @brief Frees the neural network layer.
 * @details Use this function to destroy neural network layer. Fails if layer is
 * owned by a model.
 * @since_tizen 6.0
 * @param[in] layer The NNTrainer layer handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_layer_destroy(ml_train_layer_h layer);

/**
 * @brief Sets the neural network layer Property.
 * @details Use this function to set neural network layer Property.
 * @since_tizen 6.0
 * @param[in] layer The NNTrainer layer handle.
 * @param[in]  ... Property values with NULL for termination.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 *
 * Here is an example of the usage of this function:
 * @code
 * int status;
 * ml_train_layer_h handle;
 *
 * status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
 * if (status != ML_ERROR_NONE) {
 *    // Handle error case
 *    return status;
 * }
 *
 * // Many of these hyperparmeters are optional
 * status = ml_train_layer_set_property(handle, "input_shape=1:1:6270",
 *      "unit=10", "bias_initializer=zeros", "activation=sigmoid",
 *      "weight_regularizer=l2_norm", "weight_initializer=he_uniform", NULL);
 * if (status != ML_ERROR_NONE) {
 *    // Handle error case
 *    ml_train_layer_destroy(handle);
 *    return status;
 * }
 *
 * status = ml_train_layer_destroy(handle);
 * if (status != ML_ERROR_NONE) {
 *    // Handle error case
 *    return status;
 * }
 * @endcode
 */
int ml_train_layer_set_property(ml_train_layer_h layer, ...);

/**
 * @brief Creates a neural network optimizer.
 * @details Use this function to create neural network optimizer. If not set to
 * a model, @a optimizer should be released using ml_train_optimizer_destroy().
 * If set to a model, @a optimizer is available until model is released.
 * @since_tizen 6.0
 * @remarks If the function succeeds, @a optimizer must be released using
 * ml_train_optimizer_destroy(), if not set to a model. If set to a model, @a
 * optimizer is available until the model is released.
 * @param[out] optimizer The NNTrainer optimizer handle.
 * @param[in] type The NNTrainer optimizer type.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_optimizer_create(ml_train_optimizer_h *optimizer,
                              ml_train_optimizer_type_e type);

/**
 * @brief Frees the neural network optimizer.
 * @details Use this function to destroy neural network optimizer. Fails if
 * optimizer is owned by a model.
 * @since_tizen 6.0
 * @param[in] optimizer The NNTrainer optimizer handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_optimizer_destroy(ml_train_optimizer_h optimizer);

/**
 * @brief Sets the neural network optimizer property.
 * @details Use this function to set neural network optimizer property.
 * @since_tizen 6.0
 * @param[in] optimizer The NNTrainer optimizer handle.
 * @param[in]  ... Property values with NULL for termination.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_optimizer_set_property(ml_train_optimizer_h optimizer, ...);

/**
 * @brief Creates a dataset with generators to feed to a neural network.
 * @details Use this function to create a neural network dataset using
 * generators. The generators will provide data representing a single input
 * batch. When setting this dataset to a model, the data generated by the
 * generators should match the input and the label shape for the model.
 * @since_tizen 6.0
 * @remarks If the function succeeds, @a dataset must be released using
 * ml_train_dataset_destroy(), if not set to a model. If set to a model, @a
 * dataset is available until the model is released.
 *
 * @param[out] dataset The NNTrainer dataset handle from the given description.
 * If not set to a model, @a dataset should be released using
 * ml_train_dataset_destroy(). If set to a model, @a dataset is available until
 * model is released.
 * @param[in] train_cb The dataset generator for training.
 * @param[in] valid_cb The dataset generator for validating. Can be null.
 * @param[in] test_cb The dataset generator for testing. Can be null.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_create_with_generator(ml_train_dataset_h *dataset,
                                           ml_train_datagen_cb train_cb,
                                           ml_train_datagen_cb valid_cb,
                                           ml_train_datagen_cb test_cb);

/**
 * @brief Creates a dataset with files to feed to a neural network.
 * @details Use this function to create a neural network dataset using
 * files.
 * @since_tizen 6.0
 * @param[out] dataset The NNTrainer dataset handle from the given description.
 * If not set to a model, @a dataset should be released using
 * ml_train_dataset_destroy(). If set to a model, @a dataset is available until
 * model is released.
 * @param[in] train_file The dataset file for training.
 * @param[in] valid_file The dataset file for validating. Can be null.
 * @param[in] test_file The dataset file for testing. Can be null.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_create_with_file(ml_train_dataset_h *dataset,
                                      const char *train_file,
                                      const char *valid_file,
                                      const char *test_file);

/**
 * @brief Frees the neural network dataset.
 * @details Use this function to destroy dataset. Fails if dataset is owned by a
 * model.
 * @since_tizen 6.0
 * @param[in] dataset The NNTrainer dataset handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_destroy(ml_train_dataset_h dataset);

/**
 * @brief Sets the neural network dataset property.
 * @details Use this function to set dataset property.
 * @since_tizen 6.0
 * @param[in] dataset The NNTrainer dataset handle.
 * @param[in]  ... Property values with NULL for termination.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_set_property(ml_train_dataset_h dataset, ...);

/**
 * @}
 */
#ifdef __cplusplus
}

#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNTRAINER_H__ */
