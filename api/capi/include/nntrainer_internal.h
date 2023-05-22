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
 */

#ifndef __TIZEN_MACHINELEARNING_NNTRAINER_INTERNAL_H__
#define __TIZEN_MACHINELEARNING_NNTRAINER_INTERNAL_H__

#include <array>
#include <mutex>
#include <string>
#include <unordered_map>

#include <nntrainer.h>

#include <dataset.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <tensor_dim.h>

#include <nntrainer_log.h>

/**
 * @brief Magic number of nntrainer.
 * @since_tizen 6.0
 */
#define ML_NNTRAINER_MAGIC 0x777F888F

/* Tizen ML feature */
#if defined(__TIZEN__)

/**
 * @brief Define enum for ML feature.
 * @since_tizen 7.0
 */
typedef enum {
  ML_FEATURE = 0,       /**< default option for ml feature */
  ML_FEATURE_INFERENCE, /**< inference option for ml feature */
  ML_FEATURE_TRAINING,  /**< training option for ml feature */
  ML_FEATURE_SERVICE,   /**< service option for ml feature */
  ML_FEATURE_MAX        /**< max option for ml feature */
} ml_feature_e;

/**
 * @brief Define enum for ML feature state.
 * @since_tizen 6.0
 */
typedef enum {
  NOT_CHECKED_YET = -1, /**< not checked option for feature state */
  NOT_SUPPORTED = 0,    /**< not supported option for feature state */
  SUPPORTED = 1         /**< supported option for feature state */
} feature_state_t;

#if defined(__FEATURE_CHECK_SUPPORT__)
/**
 * @brief Check feature state if it is supported.
 * @since_tizen 6.0
 * @return Error type
 */
#define check_feature_state()                         \
  do {                                                \
    int feature_ret = ml_tizen_get_feature_enabled(); \
    if (ML_ERROR_NONE != feature_ret)                 \
      return feature_ret;                             \
  } while (0);

/**
 * @brief Set feature state if it is supported.
 * @since_tizen 6.0
 */
#define set_feature_state(...) ml_train_tizen_set_feature_state(__VA_ARGS__)
#else /** __FEATURE_CHECK_SUPPORT__ @sicne_tizen 6.0 */
#define check_feature_state()
#define set_feature_state(...)
#endif /* __FEATURE_CHECK_SUPPORT__ */

#else /* __TIZEN__ */
#define check_feature_state()
#define set_feature_state(...)
#endif /* __TIZEN__ */

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Struct to wrap neural network layer for the API.
 * @since_tizen 6.0
 * @note model mutex must be locked before layer lock, if model lock is needed
 */
typedef struct {
  uint magic;                              /**< magic number */
  std::shared_ptr<ml::train::Layer> layer; /**< layer object */
  bool in_use;                             /**< in_use flag */
  std::mutex m;                            /**< mutex for the optimizer */
} ml_train_layer;

/**
 * @brief Struct to wrap learning rate scheduler for the API
 * @note optimizer mutex must be locked before learning rate scheduler lock, if
 * optimizer lock is needed
 */
typedef struct {
  uint magic;
  std::shared_ptr<ml::train::LearningRateScheduler> lr_scheduler;
  bool in_use;
  std::mutex m;
} ml_train_lr_scheduler;

/**
 * @brief Struct to wrap neural network optimizer for the API
 * @note model mutex must be locked before optimizer lock, if model lock is
 * needed
 */
typedef struct {
  uint magic;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  ml_train_lr_scheduler *lr_scheduler;
  bool in_use;
  std::mutex m;
} ml_train_optimizer;

/**
 * @brief Struct to wrap data buffer for the API.
 * @since_tizen 6.0
 * @note model mutex must be locked before dataset lock, if model lock is needed
 */
typedef struct {
  uint magic; /**< magic number */
  std::array<std::shared_ptr<ml::train::Dataset>, 3>
    dataset;    /**< dataset object */
  bool in_use;  /**< in_use flag */
  std::mutex m; /**< mutex for the dataset */
} ml_train_dataset;

/**
 * @brief Struct to wrap neural network model for the API.
 * @since_tizen 6.0
 */
typedef struct {
  uint magic;                              /**< magic number */
  std::shared_ptr<ml::train::Model> model; /**< model object */
  std::unordered_map<std::string, ml_train_layer *>
    layers_map;                  /**< layers map */
  ml_train_optimizer *optimizer; /**< optimizer object */
  ml_train_dataset *dataset;     /**< dataset object */
  std::mutex m;                  /**< mutex for the model */
} ml_train_model;

/**
 * @brief Check validity of handle to be not NULL.
 * @since_tizen 6.0
 */
#define ML_TRAIN_VERIFY_VALID_HANDLE(obj_h)                     \
  do {                                                          \
    if (!obj_h) {                                               \
      ml_loge("Error: Invalid Parameter : argument is empty."); \
      return ML_ERROR_INVALID_PARAMETER;                        \
    }                                                           \
  } while (0)

/**
 * @brief     Check validity of the user passed arguments
 */
#define ML_TRAIN_GET_VALID_HANDLE(obj, obj_h, obj_type, obj_name)     \
  do {                                                                \
    obj = (obj_type *)obj_h;                                          \
    if (obj->magic != ML_NNTRAINER_MAGIC) {                           \
      ml_loge("Error: Invalid Parameter : %s is invalid.", obj_name); \
      return ML_ERROR_INVALID_PARAMETER;                              \
    }                                                                 \
  } while (0)

/**
 * @brief Get handle to lock the passed object.
 * @since_tizen 6.0
 * @note Check validity of the user passed arguments and lock the object.
 */
#define ML_TRAIN_GET_VALID_HANDLE_LOCKED(obj, obj_h, obj_type, obj_name) \
  do {                                                                   \
    ML_TRAIN_VERIFY_VALID_HANDLE(obj_h);                                 \
    std::lock_guard<std::mutex> ml_train_lock(GLOCK);                    \
    ML_TRAIN_GET_VALID_HANDLE(obj, obj_h, obj_type, obj_name);           \
    obj->m.lock();                                                       \
  } while (0)

/**
 * @brief     Check validity of the user passed arguments, reset magic if in use
 * and lock the object
 */
#define ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(obj, obj_h, obj_type, obj_name) \
  do {                                                                         \
    ML_TRAIN_VERIFY_VALID_HANDLE(obj_h);                                       \
    std::lock_guard<std::mutex> ml_train_lock(GLOCK);                          \
    ML_TRAIN_GET_VALID_HANDLE(obj, obj_h, obj_type, obj_name);                 \
    if (!obj->in_use)                                                          \
      obj->magic = 0;                                                          \
    obj->m.lock();                                                             \
  } while (0)

/**
 * @brief     Reset object magic
 */
#define ML_TRAIN_RESET_VALIDATED_HANDLE(obj)          \
  do {                                                \
    std::lock_guard<std::mutex> ml_train_lock(GLOCK); \
    obj->magic = 0;                                   \
  } while (0)

/**
 * @brief     Check validity of passed model and lock the object.
 * @since_tizen 6.0
 */
#define ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model) \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nnmodel, model, ml_train_model, "model")

/**
 * @brief     Check validity of passed model, reset magic and lock the object.
 * @since_tizen 6.0
 */
#define ML_TRAIN_GET_VALID_MODEL_LOCKED_RESET(nnmodel, model)           \
  do {                                                                  \
    ML_TRAIN_VERIFY_VALID_HANDLE(model);                                \
    std::lock_guard<std::mutex> ml_train_lock(GLOCK);                   \
    ML_TRAIN_GET_VALID_HANDLE(nnmodel, model, ml_train_model, "model"); \
    nnmodel->magic = 0;                                                 \
    nnmodel->m.lock();                                                  \
  } while (0)

/**
 * @brief     Check validity of passed layer and lock the object.
 * @since_tizen 6.0
 */
#define ML_TRAIN_GET_VALID_LAYER_LOCKED(nnlayer, layer) \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nnlayer, layer, ml_train_layer, "layer")

/**
 * @brief     Check validity of passed layer, reset magic and lock the object.
 * @since_tizen 6.0
 */
#define ML_TRAIN_GET_VALID_LAYER_LOCKED_RESET(nnlayer, layer)            \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(nnlayer, layer, ml_train_layer, \
                                         "layer")

/**
 * @brief     Check validity of passed optimizer and lock the object.
 * @since_tizen 6.0
 */
#define ML_TRAIN_GET_VALID_OPT_LOCKED(nnopt, opt) \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nnopt, opt, ml_train_optimizer, "optimizer")

/**
 * @brief     Check validity of passed optimizer, reset magic and lock the
 * object.
 * @since_tizen 6.0
 */
#define ML_TRAIN_GET_VALID_OPT_LOCKED_RESET(nnopt, opt)                  \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(nnopt, opt, ml_train_optimizer, \
                                         "optimizer")

/**
 * @brief     Check validity of passed lr_scheduler and lock the object
 */
#define ML_TRAIN_GET_VALID_LR_SCHEDULER_LOCKED(nnlrscheduler, lrscheduler) \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nnlrscheduler, lrscheduler,             \
                                   ml_train_lr_scheduler, "lr_scheduler")

/**
 * @brief     Check validity of passed lr_scheduler, reset magic and lock the
 * object
 */
#define ML_TRAIN_GET_VALID_LR_SCHEDULER_LOCKED_RESET(nnlrscheduler, \
                                                     lrscheduler)   \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(                           \
    nnlrscheduler, lrscheduler, ml_train_lr_scheduler, "lr_scheduler")

/**
 * @brief     Check validity of passed dataset and lock the object
 */
#define ML_TRAIN_GET_VALID_DATASET_LOCKED(nndataset, dataset)            \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nndataset, dataset, ml_train_dataset, \
                                   "dataset")

/**
 * @brief     Check validity of passed dataset, reset magic and lock the object.
 * @since_tizen 6.0
 */
#define ML_TRAIN_GET_VALID_DATASET_LOCKED_RESET(nndataset, dataset)            \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(nndataset, dataset, ml_train_dataset, \
                                         "dataset")

/**
 * @brief Get all neural network layer names from the model.
 * @details Use this function to get already created Neural Network Layer names.
 * This can be used to obtain layers when model is defined with ini file.
 * @since_tizen 6.x
 * @note The caller must free the list of the layer names.
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[out] layers_name List of names of layers in the model ended with NULL.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_get_all_layer_names(ml_train_model_h model,
                                       const char **layers_name[]);

/**
 * @brief Callback function to notify completion of training of the model.
 * @since_tizen 6.0
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
 * @note If length of @a input_layer_names is more than 1, the layer to be
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

/**
 * @brief Compiles and finalizes the neural network model with single param.
 * @details Use this function to initialize neural network model. Various
 * @since_tizen 7.0
 * hyperparameter before compile the model can be set. Once compiled,
 * any modification to the properties of model or layers/dataset/optimizer in
 * the model will be restricted. Further, addition of layers or changing the
 * optimizer/dataset of the model will not be permitted.
 * API to solve va_list issue of Dllimport of C# interop.
 * The input format of single_param must be 'key = value' format, and it
 * received as shown in the example below. delimiter is '|'. e.g)
 * ml_train_model_compile_with_single_param(model, "loss=cross|batch_size=9")
 * @param[in] model The NNTrainer model handle.
 * @param[in] single_param hyperparameters for compiling the model
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_compile_with_single_param(ml_train_model_h model,
                                             const char *single_param);

/**
 * @brief Trains the neural network model with single param.
 * @details Use this function to train the compiled neural network model with
 * the passed training hyperparameters. This function will return once the
 * training, along with requested validation and testing, is completed.
 * @since_tizen 7.0
 * API to solve va_list issue of Dllimport of C# interop.
 * The input format of single_param must be 'key = value' format, and it
 * received as shown in the example below. delimiter is '|'. e.g)
 * ml_train_model_run_with_single_param(model, "epochs=2|batch_size=16")
 * @param[in] model The NNTrainer model handle.
 * @param[in] single_param Hyperparameters for train model.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_run_with_single_param(ml_train_model_h model,
                                         const char *single_param);

/**
 * @brief Sets the neural network layer Property with single param.
 * @details Use this function to set neural network layer Property.
 * @since_tizen 7.0
 * API to solve va_list issue of Dllimport of C# interop.
 * The input format of single_param must be 'key = value' format, and it
 * received as shown in the example below. delimiter is '|'. e.g)
 * ml_train_layer_set_property_with_single_param(layer,
 * "unit=10|activation=softmax")
 * @param[in] layer The NNTrainer layer handle.
 * @param[in] single_param Property values.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_layer_set_property_with_single_param(ml_train_layer_h layer,
                                                  const char *single_param);

/**
 * @brief Sets the neural network optimizer property with single param.
 * @details Use this function to set neural network optimizer property.
 * @since_tizen 7.0
 * API to solve va_list issue of Dllimport of C# interop.
 * The input format of single_param must be 'key = value' format, and it
 * received as shown in the example below. delimiter is '|'. e.g)
 * ml_train_optimizer_set_property_with_single_param(optimizer,
 * "beta1=0.002 | beta2=0.001 | epsilon=1e-7");
 * @param[in] optimizer The NNTrainer optimizer handle.
 * @param[in] single_param Property values.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_optimizer_set_property_with_single_param(
  ml_train_optimizer_h optimizer, const char *single_param);

/**
 * @brief Sets the learning rate scheduler property with single param.
 * @details Use this function to set learning rate scheduler property.
 * @since_tizen 7.5
 * API to solve va_list issue of Dllimport of C# interop.
 * The input format of single_param must be 'key = value' format, and it
 * received as shown in the example below. delimiter is '|'. e.g)
 * ml_train_lr_scheduler_set_property_with_single_param(lr_scheduler,
 * "learning_rate=0.01 | decay_rate=0.5 | decay_steps=1000");
 * @param[in] lr_scheduler The learning rate scheduler handle.
 * @param[in] single_param Property values.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_lr_scheduler_set_property_with_single_param(
  ml_train_lr_scheduler_h lr_scheduler, const char *single_param);

/**
 * @brief Sets the neural network dataset property with single param.
 * @details Use this function to set dataset property for a specific mode.
 * API to solve va_list issue of Dllimport of C# interop.
 * The input format of single_param must be 'key = value' format, and it
 * received as shown in the example below. delimiter is '|'. e.g)
 * ml_train_dataset_set_property_for_mode_with_single_param(dataset,
 * ML_TRAIN_DATASET_MODE_TEST, "key1=value2 | key1=value2");
 * @since_tizen 7.0
 * @param[in] dataset The NNTrainer dataset handle.
 * @param[in] mode The mode to set the property.
 * @param[in] single_param Property values.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_dataset_set_property_for_mode_with_single_param(
  ml_train_dataset_h dataset, ml_train_dataset_mode_e mode,
  const char *single_param);

#if defined(__TIZEN__)
/**
 * @brief Checks whether machine_learning.training feature is enabled or not.
 * @since_tizen 6.0
 * @return flag to indicate whether the feature is enabled or not.
 */
int ml_tizen_get_feature_enabled(void);

/**
 * @brief Set the feature status of machine_learning.training.
 * This is only used for Unit test.
 * @since_tizen 7.0
 * @param[in] feature The feature to be set.
 * @param[in] state The state to be set.
 */
void ml_train_tizen_set_feature_state(ml_feature_e feature,
                                      feature_state_t state);
#endif /* __TIZEN__ */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __TIZEN_MACHINELEARNING_NNTRAINER_INTERNAL_H__ */
