// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
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
#include <cairo.h>
#include <cairo/cairo-evas-gl.h>
#include <dlog.h>
#include <efl_extension.h>
#include <nnstreamer.h>
#include <nntrainer.h>
#include <pthread.h>
#include <tizen.h>

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "nntrainer-example-custom-shortcut"

#define EDJ_PATH "edje/main.edj"
#define TRAIN_SET_PATH "trainingSet.dat"

#define MAX_TRIES 5

typedef enum _DRAW_MODE {
  INFER = 0,
  TRAIN_SMILE,
  TRAIN_SAD,
} DRAW_MODE;
typedef struct appdata {
  Evas_Object *win;
  Evas_Object *conform;
  Evas_Object *label;
  Evas_Object *naviframe;
  Elm_Object_Item *nf_it;
  Eext_Circle_Surface *circle_nf;
  Elm_Object_Item *home;
  Evas_Object *layout;

  char edj_path[PATH_MAX];

  /**< drawing related */
  Evas_Object *canvas;   /**< image object that cairo surface flushes to */
  unsigned char *pixels; /**< actual pixel data */
  Evas_Coord x_offset;   /**< x starting point of canvas position */
  Evas_Coord y_offset;   /**< y starting point of canvas position */
  int width;             /**< width of the canvas */
  int height;            /**< height of the canvas */

  cairo_surface_t *cr_surface; /**< cairo surface for the canvas */
  cairo_t *cr;                 /**< cairo engine for the canvas */
  DRAW_MODE mode;              /**< drawing mode of current canvas */
  int tries;                   /**< tells how many data has been labeled */

  /**< ML related */
  ml_pipeline_h pipeline;       /**< handle of feature extractor */
  ml_pipeline_sink_h pipe_sink; /**< sink of pipeline */
  char pipe_dst[PATH_MAX];      /**< destination path where to save */
  pthread_mutex_t pipe_lock; /**< ensures that only one pipe runs at a time */
  pthread_cond_t pipe_cond;  /**< pipe condition to block at a point */
} appdata_s;

/**
 * @brief separate route path from route
 * @note this function copies @a source, and change ':' to '\0' and return start
 * of the route and data accordingly
 * @param[in] source raw source that comes like "route_target:data\0"
 * @param[out] route route part of the string, free after use
 * @param[out] data returns pointer of raw source, freed when route is freed. do
 * not free.
 * @param[out] length of data len
 * @retval 0 if no error
 */
int data_parse_route(const char *source, char **route, char **data);

/**
 * @brief get full resource path for given file.
 * @param[in] file relative file path from resource path
 * @param[out] full_path path of the output file
 * @param[in] shared true if resource is in shared/res
 * @retval APP_ERROR_NONE if no error
 */
int data_get_resource_path(const char *file, char *full_path, bool shared);

/**
 * @brief extract data feature from given model.
 * @param[in] ad appdata
 * @param[in] dst state the name of the data set
 * @param[in] append decide whether to append to the exisiting file
 *
 * This function runs a mobilnetv2 last layer detached and saves an output
 * vector. input for this model is png file drawn to the canvas(stored in
 * appdata) output can be pased to nntrainer and used.
 */
int data_extract_feature(appdata_s *ad, const char *dst, bool append);

/**
 * @brief nntrainer training model
 */
void data_train_model();

#if !defined(PACKAGE)
#define PACKAGE "org.example.nntrainer-example-custom-shortcut"
#endif

#if !defined(_D)
#define LOG_D(fmt, arg...)                                                 \
  dlog_print(DLOG_DEBUG, LOG_TAG, "[%s:%d] " fmt "\n", __func__, __LINE__, \
             ##arg)
#endif

#if !defined(_I)
#define LOG_I(fmt, arg...) \
  dlog_print(DLOG_INFO, LOG_TAG, "[%s:%d] " fmt "\n", __func__, __LINE__, ##arg)
#endif

#if !defined(_W)
#define LOG_W(fmt, arg...) \
  dlog_print(DLOG_WARN, LOG_TAG, "[%s:%d] " fmt "\n", __func__, __LINE__, ##arg)
#endif

#if !defined(_E)
#define LOG_E(fmt, arg...)                                                 \
  dlog_print(DLOG_ERROR, LOG_TAG, "[%s:%d] " fmt "\n", __func__, __LINE__, \
             ##arg)
#endif

#endif /* __nntrainer_example_custom_shortcut_data_H__ */
