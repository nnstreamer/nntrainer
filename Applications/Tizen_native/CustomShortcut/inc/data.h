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
#include <efl_extension.h>

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "nntrainer-example-custom-shortcut"

#define EDJ_PATH "edje/main.edj"

typedef struct widget_instance_data {
  Evas_Object *win;
  Evas_Object *conform;
  Evas_Object *label;
  Evas_Object *naviframe;
  Eext_Circle_Surface *circle_nf;
  Evas_Object *layout;
  char edj_path[PATH_MAX];
} widget_instance_data_s;

#endif /* __nntrainer_example_custom_shortcut_data_H__ */
