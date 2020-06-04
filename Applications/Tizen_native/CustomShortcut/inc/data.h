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
#include <app.h>
#include <dlog.h>
#include <efl_extension.h>
#include <tizen.h>

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "nntrainer-example-custom-shortcut"

#define EDJ_PATH "edje/main.edj"

typedef struct appdata {
  Evas_Object *win;
  Evas_Object *conform;
  Evas_Object *label;
  Evas_Object *naviframe;
  Eext_Circle_Surface *circle_nf;
  Elm_Object_Item *home;
  Evas_Object *layout;
  char edj_path[PATH_MAX];
} appdata_s;

#if !defined(PACKAGE)
#define PACKAGE "org.example.nntrainer-example-custom-shortcut"
#endif

#endif /* __nntrainer_example_custom_shortcut_data_H__ */
