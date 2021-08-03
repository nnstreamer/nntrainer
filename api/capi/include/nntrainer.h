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
#include <nnstreamer.h>
#include <nntrainer-api-common.h>

#ifndef TIZEN_DEPRECATED_API
/**
 * @brief Tizen Deprecated API Macro.
 * @since_tizen 6.5
 *
 */
#define TIZEN_DEPRECATED_API \
  __attribute__((__visibility__("default"), deprecated))
#endif

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
 * @brief Gets input tensors information information of the model.
 * @details Use this function to get input tensors information of the model.
 * destroy @a info with @c ml_tensors_info_destroy() after use.
 * @since_tizen 6.5
 * @remarks @a model must be compiled before calling this function.
 * @remarks The returned @a info is newly created so it does not reflect future
 * changes in the model.
 * @remarks On returning error, info must shall not be destroyed with @c
 * ml_tensors_info_destory()
 *
 * @param[in] model The NNTrainer model handle.
 * @param[out] info The tensors information handle.
 * @return @c 0 on successs. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_train_model_get_input_tensors_info(ml_train_model_h model,
                                          ml_tensors_info_h *info);

/**
 * @brief Gets output tensors information information of the model.
 * @details Use this function to get output tensors information of the model.
 * destroy @a info with @c ml_tensors_info_destroy() after use.
 * @since_tizen 6.5
 * @remarks @a model must be compiled before calling this function.
 * @remarks the returned @a info is newly created so it does not reflect future
 * changes in the model
 * @remarks On returning error, info must shall not be destroyed with @c
 * ml_tensors_info_destory()
 *
 * @param[in] model The NNTrainer model handle.
 * @param[out] info The tensors information handle.
 * @return @c 0 on successs. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 * @retval #ML_ERROR_OUT_OF_MEMORY Failed to allocate required memory.
 */
int ml_train_model_get_output_tensors_info(ml_train_model_h model,
                                           ml_tensors_info_h *info);

/**
 * @brief Creates a neural network layer.
 * @details Use this function to create neural network layer.
 * @since_tizen 6.0
 * @remarks If the function succeeds, @a layer must be released using
 * @c ml_train_layer_destroy(), if not added to a model. If added to a model, @a
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
 * @deprecated Deprecated since 6.5. Use ml_train_dataset_create() instead
 * @brief Creates a dataset with generators to feed to a neural network.
 * @details Use this function to create a neural network dataset using
 * generators. The generators will provide data representing a single input
 * element. When setting this dataset to a model, the data generated by the
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
int ml_train_dataset_create_with_generator(
  ml_train_dataset_h *dataset, ml_train_datagen_cb train_cb,
  ml_train_datagen_cb valid_cb,
  ml_train_datagen_cb test_cb) TIZEN_DEPRECATED_API;

/**
 * @brief Constructs the dataset.
 * @details Use this function to create a dataset.
 * @since_tizen 6.5
 * @remarks If the function succeeds, @a dataset must be released using
 * ml_train_dataset_destroy(), if not added to a model. If added to a model, @a
 * dataset is available until the model is released.
 * @param[out] dataset The NNTrainer dataset handle.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_create(ml_train_dataset_h *dataset);

/**
 * @brief Adds data generator callback to @a dataset.
 * @details Use this function to add a data generator callback which generates a
 * single element per call to the dataset.
 * @since_tizen 6.5
 * @param[in] dataset The NNTrainer dataset handle.
 * @param[in] usage The phase where this generator should be used.
 * @param[in] cb Callback to be used for the generator.
 * @param[in] user_data user_data to be fed when @a cb is being called.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_add_generator(ml_train_dataset_h dataset,
                                   ml_train_dataset_data_usage_e usage,
                                   ml_train_datagen_cb cb, void *user_data);

/**
 * @brief Adds data file to @a dataset.
 * @details Use this function to add a data file from where data is retrieved.
 * @since_tizen 6.5
 * @previliege %http://tizen.org/privilege/mediastorage is needed if @a dataset
 * is saved to media storage.
 * @previliege %http://tizen.org/privilege/externalstorage is needed if @a
 * dataset is saved to external storage.
 * @param[in] dataset The NNTrainer dataset handle.
 * @param[in] usage The phase where this file should be used.
 * @param[in] file file path.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_add_file(ml_train_dataset_h dataset,
                              ml_train_dataset_data_usage_e usage,
                              const char *file);

/**
 * @deprecated Deprecated since 6.5. Use ml_train_dataset_create() instead
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
int ml_train_dataset_create_with_file(
  ml_train_dataset_h *dataset, const char *train_file, const char *valid_file,
  const char *test_file) TIZEN_DEPRECATED_API;

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
 * @deprecated Deprecated since 6.5. Use
 * @c ml_train_dataset_set_property_for_mode() instead
 * @brief Sets the neural network dataset property.
 * @details Use this function to set dataset property.
 * @since_tizen 6.5
 * @remarks the same property is applied over train, valid, testsets that are
 * added to the @a dataset, it is recommened to use @c
 * ml_train_dataset_set_property_for_usage() instead.
 * @param[in] dataset The NNTrainer dataset handle.
 * @param[in]  ... Property values with NULL for termination.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_set_property(ml_train_dataset_h dataset,
                                  ...) TIZEN_DEPRECATED_API;

/**
 * @brief Sets the neural network dataset property.
 * @details Use this function to set dataset property for a specific usage.
 * @since_tizen 6.5
 * @param[in] dataset The NNTrainer dataset handle.
 * @param[in] usage The usage to set the property.
 * @param[in]  ... Property values with NULL for termination.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_set_property_for_usage(ml_train_dataset_h dataset,
                                            ml_train_dataset_data_usage_e usage,
                                            ...);

/**
 * @brief Saves the model.
 * @details Use this function to save the current model. @a format.
 * describes various formats in which various selections of the
 * parameters of the models can be saved. Some formats may save
 * parameters required for training. Some other formats may save model
 * configurations. Unless stated otherwise, @c ml_train_model_compile() has to
 * be called upon the @a model before calling this function.
 * @since_tizen 6.5
 *
 * @param[in] model The NNTrainer model handle to save.
 * @param[in] file_path File path to save the file.
 * @param[in] format Format flag to determine which format should be used to
 * save.
 * @return @c 0 on success, Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER The given @a file_path is
 * invalid or taken, or @a model is not compiled.
 * @see @c ml_train_model_format_e to check which part of the model is
 * saved.
 */
int ml_train_model_save(ml_train_model_h model, const char *file_path,
                        ml_train_model_format_e format);

/**
 * @brief Loads the model.
 * @details Use this function to load the current model. @a format
 * describes various formats in which various selections of the
 * parameters of the models can be loaded. Some formats may load
 * parameters required for training. Some other formats may load model
 * configurations. Unless stated otherwise, @c ml_train_model_compile() has to
 * be called upon the @a model before calling this function.
 * @since_tizen 6.5
 *
 *
 * @param[in] model The NNTrainer model handle to load.
 * @param[in] file_path File path to load the file.
 * @param[in] format Format flag to determine which format should be used to
 * load.
 * @return @c 0 on success, Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER The given @a file_path is
 * invalid or @a model is not in valid state to load.
 * @see @c ml_train_model_format_e to check which part of the model is
 * loaded.
 */
int ml_train_model_load(ml_train_model_h model, const char *file_path,
                        ml_train_model_format_e format);

/**
 * @}
 */
#ifdef __cplusplus
}

#endif /* __cplusplus */
#endif /* __TIZEN_MACHINELEARNING_NNTRAINER_H__ */
