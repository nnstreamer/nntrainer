/**
 * @file main.h
 * @date 14 May 2020
 * @brief TIZEN Native Example App main entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#ifndef __nntrainer_example_custom_shortcut_H__
#define __nntrainer_example_custom_shortcut_H__

#include <Elementary.h>
#include <dlog.h>
#include <tizen.h>
#include <widget_app.h>
#include <widget_app_efl.h>
#include "data.h"

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "nntrainer-example-custom-shortcut"

#define EDJ_PATH "edje/main.edj"

#if !defined(PACKAGE)
#define PACKAGE "org.example.nntrainer-example-custom-shortcut"
#endif

#endif /* __nntrainer_example_custom_shortcut_H__ */
