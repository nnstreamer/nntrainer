// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file view.h
 * @date 15 May 2020
 * @brief TIZEN Native Example App view entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __nntrainer_example_custom_shortcut_view_H__
#define __nntrainer_example_custom_shortcut_view_H__

#include "data.h"
#include <Elementary.h>
#include <Evas_GL.h>
#include <dlog.h>
#include <efl_extension.h>

/**
 * @brief initiate window and conformant.
 * @param[in] ad appdata of the app
 * @param[in] w width
 * @param[in] h height
 * @retval #APP_ERROR_*
 */
int view_init(appdata_s *ad);

/**
 * @brief creates layout from edj
 * @param[in/out] ad appdata of the app
 * @param[in] group_name name of the layout to be pushed to main naviframe.
 */
int view_routes_to(appdata_s *ad, const char *group_name);

/**
 * @brief callback function to update training result
 * @param[in] data user data
 * @param[in] buffer arrays of null terminated characters
 * @param[in] nbytes max length of the buffer
 */
void view_update_result_cb(void *data, void *buffer, unsigned int nbytes);

#endif /* __nntrainer_example_custom_shortcut_view_H__ */
