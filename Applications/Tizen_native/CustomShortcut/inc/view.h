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
 * @retval #APP_ERROR_*
 */
int view_init(appdata_s *ad);

/**
 * @brief pop an item from the naviframe if empty, terminate the app
 *
 * @param ad appdata
 */
void view_pop_naviframe(appdata_s *ad);

/**
 * @brief initiate canvas
 *
 * @param[in] ad appdata
 * @return int APP_DATA_NONE when no error
 */
int view_init_canvas(appdata_s *ad);

/**
 * @brief creates layout from edj and push it to the naviframe
 * @param[in/out] ad appdata of the app
 * @param[in] path name of the layout to be pushed to main naviframe.
 */
int view_routes_to(appdata_s *ad, const char *path);

/**
 * @brief set canvas clean and update related labels
 *
 * @param ad[in] appdata
 */
void view_set_canvas_clean(appdata_s *ad);

/**
 * @brief callback function to update training result
 * @param[in] data user data
 * @param[in] buffer arrays of null terminated characters
 * @param[in] nbytes max length of the buffer
 */
void view_update_result_cb(void *data, void *buffer, unsigned int nbytes);

#endif /* __nntrainer_example_custom_shortcut_view_H__ */
