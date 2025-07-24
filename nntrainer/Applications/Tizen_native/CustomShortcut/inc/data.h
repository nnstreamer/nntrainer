// SPDX-License-Identifier: Apache-2.0
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
#define VALIDATION_SET_PATH "trainingSet.dat"

#define MAX_TRAIN_TRIES 10
#define MAX_TRIES 10

#define FEATURE_SIZE 62720
#define NUM_CLASS 2

#define CUT_OFF_THRESHOLD 0.85

#define EMOJI_SAD "😢"
#define EMOJI_SMILE "😊"
#define EMOJI_UNKNOWN "❓"

typedef enum MODE_ {
  MODE_INFER = 0,
  MODE_TRAIN,
} MODE;

typedef enum LABEL_ {
  LABEL_SMILE = 0,
  LABEL_FROWN,
  LABEL_UNSET,
} LABEL;

typedef struct appdata {
  Evas_Object *win;
  Evas_Object *conform;
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
  MODE mode;                   /**< draw mode for the canvas */
  LABEL label; /**< target label, if infer mode, it is answer else label */
  int tries;   /**< tells how many data has been labeled */

  /**< Feature extraction related */
  ml_pipeline_h pipeline;       /**< handle of feature extractor */
  ml_pipeline_sink_h pipe_sink; /**< sink of pipeline */
  char pipe_dst[PATH_MAX];      /**< destination path where to save */
  pthread_mutex_t pipe_lock; /**< ensures that only one pipe runs at a time */
  pthread_cond_t pipe_cond;  /**< pipe condition to block at a point */
  float probability;         /**< softmax label result of the inference */

  /**< Training related */
  pthread_t tid_writer;       /**< thread handler to run trainer */
  pthread_t tid_reader;       /**< thread handler to read train result */
  int pipe_fd[2];             /**< fd for pipe */
  double best_accuracy;       /**< stores best accuracy */
  unsigned int current_epoch; /**< current epoch */
  double train_loss;          /**< current loss */
} appdata_s;

typedef struct train_result {
  int epoch;         /**< current epoch */
  double accuracy;   /**< accuracy of validation result */
  double train_loss; /**< train loss */
  double valid_loss; /**< validation loss */
} train_result_s;

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
int util_parse_route(const char *source, char **route, char **data);

/**
 * @brief get full resource path for given file.
 * @param[in] file relative file path from resource path
 * @param[out] full_path path of the output file
 * @param[in] shared true if resource is in shared/res
 * @retval APP_ERROR_NONE if no error
 */
int util_get_resource_path(const char *file, char *full_path, bool shared);

/**
 * @brief save cairo surface to a drawing.
 * @param cr_surface cairo surface to save
 * @param dst destination name, it is save to the data path
 * @return int APP_ERROR_NONE if success
 */
int util_save_drawing(cairo_surface_t *cr_surface, const char *dst);

/**
 * @brief get emoji string from LABEL enum
 *
 * @param[in] label label allocate outside.
 * @param[out] emoji_str string emoticon from the enum. free after use
 * @return int APP_ERROR_NONE if success;
 */
int util_get_emoji(LABEL label, char **emoji_str);

/**
 * @brief handle given path_data. If data is invalid, it is essentially noop
 *
 * @param ad appdata
 * @param data path_data to be handled
 */
void data_handle_path_data(appdata_s *ad, const char *data);

/**
 * @brief update draw target from the data. currently the label is depending on
 * ad->tries and ad->mode
 *
 * @param[in] ad appdata
 * @return int APP_ERROR_NONE if success
 */
int data_update_label(appdata_s *ad);

/**
 * @brief extract data feature from given model.
 * @param[in] ad appdata
 *
 * This function runs a mobilnetv2 last layer detached and saves an output
 * vector. input for this model is png file drawn to the canvas(stored in
 * appdata) output can be pased to nntrainer and used.
 */
int data_extract_feature(appdata_s *ad);

/**
 * @brief nntrainer training model that is to run from pthread_create
 * @param[in] ad appdata.
 * @return not used.
 */
void *data_run_model(void *ad);

/**
 * @brief nntrainer update train result from data_run_model
 *
 * @param[in] ad appdata
 * @param[in] buf buffer hooked from stdout
 * @return APP_ERROR_NONE if suceess
 */
int data_update_train_progress(appdata_s *ad, const char *buf);

/**
 * @brief parse result string
 * @param[in] result result string to be parsed.
 * @param[out] train_result structured data from result string
 * @retval APP_ERROR_NONE if no error
 * @retval APP_ERROR_INVALID_PARAMETER if string can't be parsed
 *
 * result string is like:
 * #1/10 - Training Loss: 0.717496 >> [ Accuracy: 75% - Validation Loss :
 0.667001 ]
 * #10/10 - Training Loss: 0.398767 >> [ Accuracy: 75% - Validation Loss :
 0.467543 ]
 */
int util_parse_result_string(const char *src, train_result_s *train_result);

/**
 * @brief run inference with nnstreamer
 *
 * @param ad appdata
 * @return int APP_ERROR_NONE if no error
 */
int data_run_inference(appdata_s *ad);

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
