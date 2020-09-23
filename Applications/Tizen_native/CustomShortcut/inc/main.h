// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file main.h
 * @date 14 May 2020
 * @brief TIZEN Native Example App main entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __nntrainer_example_custom_shortcut_H__
#define __nntrainer_example_custom_shortcut_H__

#include <tizen.h>

#include <Elementary.h>

#include "data.h"
#include "view.h"

#if !defined(PACKAGE)
#define PACKAGE "org.example.nntrainer-example-custom-shortcut"
#endif

/**
 * @brief presenter to handle back press
 *
 * @param data appdata
 * @param obj naviframe of the app
 * @param event_info not used
 */
void presenter_on_back_button_press(void *data, Evas_Object *obj,
                                    void *event_info EINA_UNUSED);

/**
 * @brief presenter to handle routing to a page
 *
 * @param data appdata
 * @param obj not used
 * @param emission not used
 * @param source string information that has where to go.
 */
void presenter_on_routes_request(void *data, Evas_Object *obj EINA_UNUSED,
                                 const char *emission EINA_UNUSED,
                                 const char *source);

/**
 * @brief presenter to handle canvas submission in the inference
 *
 * @param data appdata
 * @param obj not used
 * @param emission not used
 * @param source not used
 */
void presenter_on_canvas_submit_inference(void *data, Evas_Object *obj,
                                          const char *emission EINA_UNUSED,
                                          const char *source EINA_UNUSED);

/**
 * @brief presenter to handle canvas submission in the training
 *
 * @param data appdata
 * @param obj not used
 * @param emission not used
 * @param source not used
 */
void presenter_on_canvas_submit_training(void *data, Evas_Object *obj,
                                         const char *emission EINA_UNUSED,
                                         const char *source EINA_UNUSED);

#endif /* __nntrainer_example_custom_shortcut_H__ */
