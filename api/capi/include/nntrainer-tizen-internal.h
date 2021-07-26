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
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
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
 * @brief Get neural network layer from the model with the given name.
 * @details Use this function to get already created Neural Network Layer. The
 * returned layer must not be deleted as it is owned by the model.
 * @since_tizen 6.x
 * @remark This works only for layers added by the API, and does not currently
 * support layers supported for layers created by the ini.
 * @remark The modification through ml_trin_layer_set_property() after compiling
 * the model by calling `ml_train_model_compile()` strictly restricted.
 * @param[in] model The NNTrainer model handler from the given description.
 * @param[in] layer_name Name of the already created layer.
 * @param[out] layer The NNTrainer Layer handler from the given description.
 * @return @c 0 on success. Otherwise a negative error value.
 * @retval #ML_ERROR_NONE Successful.
 * @retval #ML_ERROR_INVALID_PARAMETER Invalid parameter.
 */
int ml_train_model_get_layer(ml_train_model_h model, const char *layer_name,
                             ml_train_layer_h *layer);

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif // __NNTRAINER_TIZEN_INTERNAL_H__
