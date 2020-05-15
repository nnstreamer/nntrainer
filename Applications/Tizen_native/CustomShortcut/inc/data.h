/**
 * @file data.h
 * @date 15 May 2020
 * @brief TIZEN Native Example App dataentry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __nntrainer_example_custom_shortcut_data_H__
#define __nntrainer_example_custom_shortcut_data_H__

#include <Elementary.h>
#include <dlog.h>
#include <tizen.h>
#include <widget_app.h>
#include <widget_app_efl.h>

typedef struct widget_instance_data {
  Evas_Object *win;
  Evas_Object *conform;
  Evas_Object *label;
} widget_instance_data_s;

#endif /* __nntrainer_example_custom_shortcut_data_H__ */
