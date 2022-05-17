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

#ifndef __NNTRAINER_INTERNAL_H__
#define __NNTRAINER_INTERNAL_H__

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

#define ML_NNTRAINER_MAGIC 0x777F888F

/* Tizen ML feature */
#if defined(__TIZEN__)

typedef enum {
  ML_FEATURE = 0,
  ML_FEATURE_INFERENCE,
  ML_FEATURE_TRAINING,
  ML_FEATURE_SERVICE,

  ML_FEATURE_MAX
} ml_feature_e;

typedef enum {
  NOT_CHECKED_YET = -1,
  NOT_SUPPORTED = 0,
  SUPPORTED = 1
} feature_state_t;

#if defined(__FEATURE_CHECK_SUPPORT__)
#define check_feature_state()                         \
  do {                                                \
    int feature_ret = ml_tizen_get_feature_enabled(); \
    if (ML_ERROR_NONE != feature_ret)                 \
      return feature_ret;                             \
  } while (0);

#define set_feature_state(...) ml_train_tizen_set_feature_state(__VA_ARGS__)
#else /* __FEATURE_CHECK_SUPPORT__ */
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
 * @brief Struct to wrap neural network layer for the API
 * @note model mutex must be locked before layer lock, if model lock is needed
 */
typedef struct {
  uint magic;
  std::shared_ptr<ml::train::Layer> layer;
  bool in_use;
  std::mutex m;
} ml_train_layer;

/**
 * @brief Struct to wrap neural network optimizer for the API
 * @note model mutex must be locked before optimizer lock, if model lock is
 * needed
 */
typedef struct {
  uint magic;
  std::shared_ptr<ml::train::Optimizer> optimizer;
  bool in_use;
  std::mutex m;
} ml_train_optimizer;

/**
 * @brief Struct to wrap data buffer for the API
 * @note model mutex must be locked before dataset lock, if model lock is needed
 */
typedef struct {
  uint magic;
  std::array<std::shared_ptr<ml::train::Dataset>, 3> dataset;
  bool in_use;
  std::mutex m;
} ml_train_dataset;

/**
 * @brief Struct to wrap neural network model for the API
 */
typedef struct {
  uint magic;
  std::shared_ptr<ml::train::Model> model;
  std::unordered_map<std::string, ml_train_layer *> layers_map;
  ml_train_optimizer *optimizer;
  ml_train_dataset *dataset;
  std::mutex m;
} ml_train_model;

/**
 * @brief     Check validity of handle to be not NULL
 */
#define ML_TRAIN_VERIFY_VALID_HANDLE(obj_h)                     \
  do {                                                          \
    if (!obj_h) {                                               \
      ml_loge("Error: Invalid Parameter : argument is empty."); \
      return ML_ERROR_INVALID_PARAMETER;                        \
    }                                                           \
  } while (0)

/**
 * @brief     Check validity of the user passed arguments and lock the object
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
 * @brief     Check validity of the user passed arguments and lock the object
 */
#define ML_TRAIN_GET_VALID_HANDLE_LOCKED(obj, obj_h, obj_type, obj_name) \
  do {                                                                   \
    ML_TRAIN_VERIFY_VALID_HANDLE(obj_h);                                 \
    std::lock_guard<std::mutex> ml_train_lock(GLOCK);                    \
    ML_TRAIN_GET_VALID_HANDLE(obj, obj_h, obj_type, obj_name);           \
    obj->m.lock();                                                       \
  } while (0)

/**
 * @brief     Check validity of the user passed arguments and lock the object
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
 * @brief     Check validity of the user passed arguments and lock the object
 */
#define ML_TRAIN_RESET_VALIDATED_HANDLE(obj)          \
  do {                                                \
    std::lock_guard<std::mutex> ml_train_lock(GLOCK); \
    obj->magic = 0;                                   \
  } while (0)

/**
 * @brief     Check validity of passed model and lock the object
 */
#define ML_TRAIN_GET_VALID_MODEL_LOCKED(nnmodel, model) \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nnmodel, model, ml_train_model, "model")

/**
 * @brief     Check validity of passed model, reset magic and lock the object
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
 * @brief     Check validity of passed layer and lock the object
 */
#define ML_TRAIN_GET_VALID_LAYER_LOCKED(nnlayer, layer) \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nnlayer, layer, ml_train_layer, "layer")

/**
 * @brief     Check validity of passed layer, reset magic and lock the object
 */
#define ML_TRAIN_GET_VALID_LAYER_LOCKED_RESET(nnlayer, layer)            \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(nnlayer, layer, ml_train_layer, \
                                         "layer")

/**
 * @brief     Check validity of passed optimizer and lock the object
 */
#define ML_TRAIN_GET_VALID_OPT_LOCKED(nnopt, opt) \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nnopt, opt, ml_train_optimizer, "optimizer")

/**
 * @brief     Check validity of passed optimizer, reset magic and lock the
 * object
 */
#define ML_TRAIN_GET_VALID_OPT_LOCKED_RESET(nnopt, opt)                  \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(nnopt, opt, ml_train_optimizer, \
                                         "optimizer")

/**
 * @brief     Check validity of passed dataset and lock the object
 */
#define ML_TRAIN_GET_VALID_DATASET_LOCKED(nndataset, dataset)            \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED(nndataset, dataset, ml_train_dataset, \
                                   "dataset")

/**
 * @brief     Check validity of passed dataset, reset magic and lock the object
 */
#define ML_TRAIN_GET_VALID_DATASET_LOCKED_RESET(nndataset, dataset)            \
  ML_TRAIN_GET_VALID_HANDLE_LOCKED_RESET(nndataset, dataset, ml_train_dataset, \
                                         "dataset")

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

/**
 * @brief Compiles and finalizes the neural network model with the params.
 * @details Use this function to initialize neural network model. Various
 * @since_tizen 7.0
 * hyperparameter before compile the model can be set. Once compiled,
 * any modification to the properties of model or layers/dataset/optimizer in
 * the model will be restricted. Further, addition of layers or changing the
 * optimizer/dataset of the model will not be permitted.
 * @param[in] model The NNTrainer model handle.
 * @param[in] single_param hyperparameters for compiling the model
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_NOT_SUPPORTED Not supported.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 * @details API to solve va_list issue of Dllimport of C# interop.
 * The input format of single_param must be 'key = value' format, and it
 * received as shown in the example below. delimiter is '|'. e.g)
 * ml_train_model_compile_with_single_param(model, "loss = cross`batch_size =
 * 9")
 */
int ml_train_model_compile_with_single_param(ml_train_model_h model,

#if defined(__TIZEN__)
/**
 * @brief Checks whether machine_learning.training feature is enabled or not.
 */
int ml_tizen_get_feature_enabled(void);

/**
 * @brief Set the feature status of machine_learning.training.
 * This is only used for Unit test.
 */
void ml_train_tizen_set_feature_state(ml_feature_e feature,
                                      feature_state_t state);
#endif /* __TIZEN__ */

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
