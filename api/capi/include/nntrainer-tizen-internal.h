// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   nntrainer-tizen-internal.h
 * @date   30 June 2021
 * @brief  NNTrainer CAPI header for tizen interanl api.
 * @note   This header is designed to be used only in Tizen
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_TIZEN_INTERNAL_H__
#define __NNTRAINER_TIZEN_INTERNAL_H__

#include <nntrainer.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/**
 * @brief Save the model
 * @details Use this function to save the current model. @a format
 * describes various formats in which various selections of the
 * parameters of the models can be saved. Some formats may save
 * parameters required for training. Some other formats may save model
 * configurations. Unless stated otherwise, @a ml_train_model_compile() has to
 * be called upon the @a model before calling this function.
 * @see @a ml_train_model_format_e to check which part of the model is
 * saved
 * @note This function overrides the existing file without any notification.
 *
 * @since_tizen 6.5
 * @param[in] model The NNTrainer model handle to save
 * @param[in] file_path File path to save the file.
 * @param[in] format Format flag to determine which format should be used to
 * save
 * @return @c 0 on success, Otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER The given @a file_path is
 * invalid or @a model is not compiled.
 */
int ml_train_model_save(ml_train_model_h model, const char *file_path,
                        ml_train_model_format_e format);

/**
 * @brief Load the model
 * @details Use this function to load the current model. @a format
 * describes various formats in which various selections of the
 * parameters of the models can be loaded. Some formats may load
 * parameters required for training. Some other formats may load model
 * configurations. Unless stated otherwise, @a ml_train_model_compile() has to
 * be called upon the @a model before calling this function.
 *
 * @see @a ml_train_model_format_e to check which part of the model is
 * loaded
 *
 * @since_tizen 6.5
 * @param[in] model The NNTrainer model handle to load.
 * @param[in] file_path File path to load the file.
 * @param[in] format Format flag to determine which format should be used to
 * loaded
 * @return @c 0 on success, Otherwise a negative error value
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER The given @a file_path is
 * invalid or @a model is not in valid state to load.
 */
int ml_train_model_load(ml_train_model_h model, const char *file_path,
                        ml_train_model_format_e format);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif // __NNTRAINER_TIZEN_INTERNAL_H__
