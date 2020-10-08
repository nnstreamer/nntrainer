// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file data.c
 * @date 21 Jul 2020
 * @brief TIZEN Native Example App data entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */
#include <image_util.h>
#include <regex.h>
#include <stdio.h>
#include <string.h>

#include "data.h"

int util_parse_route(const char *source, char **route, char **data) {
  char *dst = strdup(source);
  const char sep = ':';
  char *i;
  bool find_data = false;

  if (route == NULL || data == NULL) {
    free(dst);
    return APP_ERROR_INVALID_PARAMETER;
  }

  *route = dst;

  for (i = dst; *i != '\0'; ++i) {
    if (*i == sep) {
      *i = '\0';
      *data = i + 1;
      find_data = true;
      break;
    }
  }

  if (!find_data) {
    *data = NULL;
  }

  return APP_ERROR_NONE;
}

int util_get_resource_path(const char *file, char *full_path, bool shared) {
  char *root_path;
  if (shared) {
    root_path = app_get_shared_resource_path();
  } else {
    root_path = app_get_resource_path();
  }

  if (root_path == NULL) {
    LOG_E("failed to get resource path");
    return APP_ERROR_INVALID_PARAMETER;
  }

  if (full_path == NULL) {
    LOG_E("full_path is null");
    free(root_path);
    return APP_ERROR_INVALID_PARAMETER;
  }

  snprintf(full_path, PATH_MAX, "%s%s", root_path, file);
  LOG_D("resource path: %s", full_path);
  free(root_path);

  return APP_ERROR_NONE;
}

int util_get_data_path(const char *file, char *full_path) {
  char *root_path;

  root_path = app_get_data_path();

  if (root_path == NULL) {
    LOG_E("failed to get data path");
    return APP_ERROR_INVALID_PARAMETER;
  }

  if (full_path == NULL) {
    LOG_E("full_path is null");
    free(root_path);
    return APP_ERROR_INVALID_PARAMETER;
  }

  snprintf(full_path, PATH_MAX, "%s%s", root_path, file);
  LOG_D("data path: %s", full_path);
  free(root_path);

  return APP_ERROR_NONE;
}

int util_save_drawing(cairo_surface_t *cr_surface, const char *dst) {
  static const image_util_colorspace_e colorspace =
    IMAGE_UTIL_COLORSPACE_ARGB8888;
  static const image_util_type_e imagetype = IMAGE_UTIL_JPEG;
  image_util_decode_h decoder;
  image_util_encode_h encoder;
  image_util_image_h image;
  int status = IMAGE_UTIL_ERROR_NONE;

  cairo_status_t cr_stat = CAIRO_STATUS_SUCCESS;
  LOG_D("start writing to jpg_path: %s ", dst);

  cr_stat = cairo_surface_write_to_png(cr_surface, dst);
  if (cr_stat != CAIRO_STATUS_SUCCESS) {
    LOG_E("failed to write cairo surface as a file reason: %d", cr_stat);
    return APP_ERROR_INVALID_PARAMETER;
  }

  status = image_util_decode_create(&decoder);
  if (status != IMAGE_UTIL_ERROR_NONE) {
    LOG_E("util decode create failed, reason: %d", status);
    return status;
  }

  status = image_util_decode_set_input_path(decoder, dst);
  if (status != IMAGE_UTIL_ERROR_NONE) {
    LOG_E("setting input buffer for decoder failed, reason: %d", status);
    goto DESTORY_DECODER;
  }

  status = image_util_decode_set_colorspace(decoder, colorspace);
  if (status != IMAGE_UTIL_ERROR_NONE) {
    LOG_E("setting colorspace for decoder failed, reason: %d", status);
    goto DESTORY_DECODER;
  }

  status = image_util_decode_run2(decoder, &image);
  if (status != IMAGE_UTIL_ERROR_NONE) {
    LOG_E("decoding image failed, reason: %d", status);
    goto DESTORY_DECODER;
  }

  status = image_util_encode_create(imagetype, &encoder);
  if (status != IMAGE_UTIL_ERROR_NONE) {
    LOG_E("creating encoder failed, reason: %d", status);
    goto DESTROY_IMAGE;
  }

  status = image_util_encode_run_to_file(encoder, image, dst);
  if (status != IMAGE_UTIL_ERROR_NONE) {
    LOG_E("encoding file failed, reason: %d", status);
  }

  image_util_encode_destroy(encoder);

DESTROY_IMAGE:
  image_util_destroy_image(image);

DESTORY_DECODER:
  image_util_decode_destroy(decoder);

  return status;
}

int util_get_emoji(LABEL label, char **emoji_str) {
  *emoji_str = (char *)malloc(sizeof(char) * 5);

  /// setting draw label and text
  switch (label) {
  case LABEL_UNSET:
    strcpy(*emoji_str, EMOJI_UNKNOWN);
    return APP_ERROR_NONE;
  case LABEL_SMILE:
    strcpy(*emoji_str, EMOJI_SMILE);
    return APP_ERROR_NONE;
  case LABEL_FROWN:
    strcpy(*emoji_str, EMOJI_SAD);
    return APP_ERROR_NONE;
  default:
    LOG_E("unreachable code");
    return APP_ERROR_INVALID_CONTEXT;
  }
  return APP_ERROR_INVALID_CONTEXT;
}

/************** data releated methods **********************/

static void on_feature_receive_(ml_tensors_data_h data,
                                const ml_tensors_info_h info, void *user_data) {
  appdata_s *ad = (appdata_s *)user_data;

  void *raw_data;
  size_t data_size, write_result;
  FILE *file;
  float label;

  pthread_mutex_lock(&ad->pipe_lock);

  /// @note data_size written here will be overriden and it is intended.
  int status = ml_tensors_data_get_tensor_data(data, 0, &raw_data, &data_size);
  if (status != ML_ERROR_NONE) {
    LOG_E("get tensor data failed %d", status);
    status = ml_tensors_info_get_tensor_size(info, -1, &data_size);
    goto CLEAN;
  }

  LOG_I("current tries %d", ad->tries);

  if (ad->tries == MAX_TRAIN_TRIES || ad->tries == 0)
    file = fopen(ad->pipe_dst, "wb+");
  else
    file = fopen(ad->pipe_dst, "ab+");

  if (file == NULL) {
    LOG_E("cannot open file");
    goto CLEAN;
  }

  if ((write_result = fwrite(raw_data, 1, data_size, file)) < 0) {
    LOG_E("write error happend");
  }

  if (write_result < data_size) {
    LOG_E("data was not fully written to file, result = %d, data_size = %d",
          write_result, data_size);
  }

  /// one-hot encoding.
  /// SMILE: 0 1
  /// FROWN: 1 0
  bool target_label = ad->label == LABEL_SMILE ? 0 : 1;
  LOG_D("writing one-hot encoded label");
  label = target_label;
  if (fwrite(&label, sizeof(float), 1, file) < 0) {
    LOG_E("write error happend");
  };

  label = !target_label;
  if (fwrite(&label, sizeof(float), 1, file) < 0) {
    LOG_E("write error happend");
  };

  LOG_I("file dst: %s size: %ld", ad->pipe_dst, ftell(file));

  if (fclose(file) < 0) {
    LOG_E("there was error closing");
    goto CLEAN;
  }

  LOG_D("using pipeline finished, destroying pipeline");

CLEAN:
  pthread_cond_signal(&ad->pipe_cond);
  pthread_mutex_unlock(&ad->pipe_lock);
}

static int run_mobilnet_pipeline_(appdata_s *ad, const char *src) {
  char pipe_description[5000];
  char model_path[PATH_MAX];

  int status = ML_ERROR_NONE;

  util_get_resource_path("mobilenetv2.tflite", model_path, false);

  LOG_D("pipe ready, starting pipeline");

  sprintf(pipe_description,
          "filesrc location=%s ! jpegdec ! "
          "videoconvert ! videoscale ! "
          "video/x-raw,width=224,height=224,format=RGB ! "
          "tensor_converter ! "
          "tensor_transform mode=arithmetic option=%s ! "
          "tensor_filter framework=tensorflow-lite model=%s ! "
          "tensor_sink name=sink",
          src, "typecast:float32,add:-127.5,div:127.5", model_path);

  LOG_D("setting inference \n pipe: %s", pipe_description);
  status = ml_pipeline_construct(pipe_description, NULL, NULL, &ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("something wrong constructing pipeline %d", status);
    return status;
  }

  status = ml_pipeline_sink_register(ad->pipeline, "sink", on_feature_receive_,
                                     (void *)ad, &ad->pipe_sink);
  if (status != ML_ERROR_NONE) {
    LOG_E("sink register failed %d", status);
    goto PIPE_DESTORY;
  }

  LOG_D("starting inference");
  status = ml_pipeline_start(ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("failed to start pipeline %d", status);
    goto SINK_UNREGISTER;
  }

  status = pthread_mutex_lock(&ad->pipe_lock);
  if (status != 0) {
    LOG_E("acquiring lock failed status: %d", status);
    goto MUTEX_UNLOCK;
  }

  pthread_cond_wait(&ad->pipe_cond, &ad->pipe_lock);

  LOG_D("stopping pipeline");
  status = ml_pipeline_stop(ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("stopping pipeline failed");
    goto MUTEX_UNLOCK;
  }

MUTEX_UNLOCK:
  pthread_mutex_unlock(&ad->pipe_lock);

SINK_UNREGISTER:
  ml_pipeline_sink_unregister(ad->pipe_sink);
  ad->pipe_sink = NULL;

PIPE_DESTORY:
  ml_pipeline_destroy(ad->pipeline);
  ad->pipeline = NULL;

  return status;
}

void data_handle_path_data(appdata_s *ad, const char *data) {
  /// handling path_data to check if it's for inference or path
  ad->label = LABEL_UNSET;

  if (data == NULL) {
    return;
  }

  if (!strcmp(data, "inference")) {
    ad->mode = MODE_INFER;
  } else if (!strcmp(data, "train")) {
    ad->mode = MODE_TRAIN;
  }
}

int data_update_label(appdata_s *ad) {
  if (ad->mode == MODE_INFER) {
    ad->label = LABEL_UNSET;
    return APP_ERROR_NONE;
  }

  switch (ad->tries % NUM_CLASS) {
  case 0:
    ad->label = LABEL_SMILE;
    return APP_ERROR_NONE;
  case 1:
    ad->label = LABEL_FROWN;
    return APP_ERROR_NONE;
  default:
    LOG_E("Given label is unknown");
    return APP_ERROR_NOT_SUPPORTED;
  }
}

int data_extract_feature(appdata_s *ad) {
  char jpeg_path[PATH_MAX];
  const char *dst =
    ad->tries < MAX_TRAIN_TRIES ? TRAIN_SET_PATH : VALIDATION_SET_PATH;
  int status = APP_ERROR_NONE;

  status = util_get_data_path("test.jpg", jpeg_path);
  if (status != APP_ERROR_NONE) {
    LOG_E("getting data path failed");
    return status;
  }

  status = util_save_drawing(ad->cr_surface, jpeg_path);
  if (status != APP_ERROR_NONE) {
    LOG_E("failed to save drawing to a file");
    return status;
  }

  util_get_data_path(dst, ad->pipe_dst);

  LOG_I("start inference to dataset: %s ", ad->pipe_dst);
  status = run_mobilnet_pipeline_(ad, jpeg_path);

  return status;
}

void *data_run_model(void *data) {
  appdata_s *ad = (appdata_s *)data;

  int status = ML_ERROR_NONE;
  int fd_stdout, fd_pipe;

  ml_train_model_h model;

  char model_conf_path[PATH_MAX];
  char label_path[PATH_MAX];
  FILE *file;

  LOG_D("redirecting stdout");
  fd_pipe = ad->pipe_fd[1];
  fd_stdout = dup(1);
  if (fd_stdout < 0) {
    LOG_E("failed to duplicate stdout");
    return NULL;
  }

  fflush(stdout);
  if (dup2(fd_pipe, 1) < 0) {
    LOG_E("failed to redirect fd pipe");
    close(fd_stdout);
    return NULL;
  }

  LOG_D("start running model");
  util_get_resource_path("model.ini", model_conf_path, false);
  util_get_data_path("label.dat", label_path);

  LOG_D("opening file");
  file = fopen(label_path, "w");
  if (file == NULL) {
    LOG_E("Error opening file");
    return NULL;
  }

  LOG_D("writing file");
  if (fputs("sad\nsmile\n\n", file) < 0) {
    LOG_E("error writing");
    fclose(file);
    return NULL;
  }

  LOG_D("closing file");
  if (fclose(file) < 0) {
    LOG_E("Error closing file");
    return NULL;
  }

  LOG_D("model conf path: %s", model_conf_path);

  status = ml_train_model_construct_with_conf(model_conf_path, &model);
  if (status != ML_ERROR_NONE) {
    LOG_E("constructing trainer model failed %d", status);
    return NULL;
  }

  status = ml_train_model_compile(model, NULL);
  if (status != ML_ERROR_NONE) {
    LOG_E("compile model failed %d", status);
    goto CLEAN_UP;
  }

  status = ml_train_model_run(model, NULL);
  if (status != ML_ERROR_NONE) {
    LOG_E("run model failed %d", status);
    goto CLEAN_UP;
  }

CLEAN_UP:
  // restore stdout
  fflush(stdout);
  dup2(fd_stdout, 1);
  close(fd_pipe);

  status = ml_train_model_destroy(model);
  if (status != ML_ERROR_NONE) {
    LOG_E("Destroying model failed %d", status);
  }
  return NULL;
}

int data_update_train_progress(appdata_s *ad, const char *buf) {
  train_result_s result;

  if (util_parse_result_string(buf, &result) != 0) {
    LOG_W("parse failed. current buffer is being ignored");
    return APP_ERROR_INVALID_PARAMETER;
  }

  if (result.accuracy > ad->best_accuracy) {
    ad->best_accuracy = result.accuracy;
  }

  ad->current_epoch = result.epoch;
  ad->train_loss = result.train_loss;

  return APP_ERROR_NONE;
}

int util_parse_result_string(const char *src, train_result_s *train_result) {
  // clang-format off
  // #10/10 - Training Loss: 0.398767 >> [ Accuracy: 75% - Validation Loss : 0.467543 ]
  // clang-format on
  int status = APP_ERROR_NONE;
  int max_len = 512;
  char buf[512];

  regex_t pattern;
  LOG_D(">>> %s", src);
  char *pattern_string =
    "#([0-9]+).*Training Loss: ([0-9]+\\.?[0-9]*)"
    ".*Accuracy: ([0-9]+\\.?[0-9]*)%.*Validation Loss : ([0-9]+\\.?[0-9]*)";
  unsigned int match_len = 5;
  regmatch_t matches[5];

  unsigned int i;

  if (strlen(src) > max_len) {
    LOG_E("source string too long");
    return APP_ERROR_INVALID_PARAMETER;
  }

  status = regcomp(&pattern, pattern_string, REG_EXTENDED);
  LOG_D(">>> %s", src);
  if (status != 0) {
    LOG_E("Could not compile regex string");
    goto CLEAN;
  }

  status = regexec(&pattern, src, match_len, matches, 0);
  switch (status) {
  case 0:
    break;
  case REG_NOMATCH:
    LOG_E("Nothing matches for given string");
    goto CLEAN;
  default:
    LOG_E("Could not excuted regex, reason: %d", status);
    goto CLEAN;
  }

  for (i = 1; i < match_len; ++i) {
    if (matches[i].rm_so == -1) {
      LOG_D("no information for idx: %d", i);
      status = APP_ERROR_INVALID_PARAMETER;
      goto CLEAN;
    }

    int len = matches[i].rm_eo - matches[i].rm_so;
    if (len < 0) {
      LOG_D("invalid length");
      status = APP_ERROR_INVALID_PARAMETER;
      goto CLEAN;
    }

    LOG_D("match start %d, match end: %d", matches[i].rm_so, matches[i].rm_eo);
    memcpy(buf, src + matches[i].rm_so, len);
    buf[len] = '\0';
    LOG_D("Match %u: %s", i, buf);

    switch (i) {
    case 1:
      train_result->epoch = atoi(buf);
      break;
    case 2:
      train_result->train_loss = atof(buf);
      break;
    case 3:
      train_result->accuracy = atof(buf);
      break;
    case 4:
      train_result->valid_loss = atof(buf);
      break;
    default:
      /// should not reach here
      LOG_D("unknown index %d", i);
      status = APP_ERROR_INVALID_PARAMETER;
      goto CLEAN;
    }
  }
CLEAN:
  regfree(&pattern);

  return status;
}

static void on_inference_end_(ml_tensors_data_h data,
                              const ml_tensors_info_h info, void *user_data) {
  appdata_s *ad = (appdata_s *)user_data;

  float *raw_data;
  size_t data_size;

  int status = pthread_mutex_lock(&ad->pipe_lock);
  if (status != 0) {
    LOG_E("acquiring lock failed %d", status);
    pthread_cond_signal(&ad->pipe_cond);
    return;
  }

  status =
    ml_tensors_data_get_tensor_data(data, 0, (void **)&raw_data, &data_size);
  if (status != ML_ERROR_NONE) {
    LOG_E("get tensor data failed: reason %s", strerror(status));
    goto RESUME;
  }

  if (data_size != sizeof(float) * NUM_CLASS) {
    LOG_E("output tensor size mismatch, %d", (int)data_size);
    goto RESUME;
  }

  /// SMILE: 0 1
  /// FROWN: 1 0
  LOG_D("\033[33mlabel: %lf %lf\033[0m", raw_data[0], raw_data[1]);

  ad->probability = raw_data[0] < raw_data[1] ? raw_data[1] : raw_data[0];
  ad->label = raw_data[0] < raw_data[1] ? LABEL_SMILE : LABEL_FROWN;

RESUME:
  status = pthread_cond_signal(&ad->pipe_cond);
  if (status != 0) {
    LOG_E("cond signal failed %d", status);
  }
  pthread_mutex_unlock(&ad->pipe_lock);
}

int run_inference_pipeline_(appdata_s *ad, const char *filesrc) {
  char pipe_description[9000];
  char tf_model_path[PATH_MAX];
  char trainer_model_path[PATH_MAX];

  int status = APP_ERROR_NONE;

  status = util_get_resource_path("mobilenetv2.tflite", tf_model_path, false);
  if (status != APP_ERROR_NONE) {
    LOG_E("error getting resource path, reason: %d", status);
    return status;
  }

  status = util_get_resource_path("model.ini", trainer_model_path, false);
  if (status != APP_ERROR_NONE) {
    LOG_E("error getting data path, reason: %d", status);
    return status;
  }

  sprintf(pipe_description,
          "filesrc location=%s ! jpegdec ! videoconvert ! "
          "videoscale ! video/x-raw,width=224,height=224,format=RGB ! "
          "tensor_converter ! "
          "tensor_transform mode=arithmetic option=%s ! "
          "tensor_filter framework=tensorflow-lite model=%s ! "
          "tensor_filter framework=nntrainer model=%s input=1280:7:7:1 "
          "inputtype=float32 output=1:%d:1:1 outputtype=float32 ! "
          "tensor_sink name=sink",
          filesrc, "typecast:float32,add:-127.5,div:127.5", tf_model_path,
          trainer_model_path, NUM_CLASS);

  LOG_D("pipe description: %s", pipe_description);
  status = ml_pipeline_construct(pipe_description, NULL, NULL, &ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("constructing pipeline failed, reason: %d", status);
    goto PIPE_UNLOCK;
  }

  status = ml_pipeline_sink_register(ad->pipeline, "sink", on_inference_end_,
                                     (void *)ad, &ad->pipe_sink);
  if (status != ML_ERROR_NONE) {
    LOG_E("sink register failed, reason: %d", status);
    goto DESTORY_PIPE;
  }

  status = ml_pipeline_start(ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("failed to start pipeline %d", status);
    goto UNREGISTER_SINK;
  }

  status = pthread_mutex_lock(&ad->pipe_lock);
  if (status != 0) {
    LOG_E("acquiring lock failed status: %d", status);
    goto UNREGISTER_SINK;
  }

  pthread_cond_wait(&ad->pipe_cond, &ad->pipe_lock);

  status = ml_pipeline_stop(ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("stopping pipeline failed");
  }

PIPE_UNLOCK:
  pthread_mutex_unlock(&ad->pipe_lock);

UNREGISTER_SINK:
  ml_pipeline_sink_unregister(ad->pipe_sink);
  ad->pipe_sink = NULL;

DESTORY_PIPE:
  ml_pipeline_destroy(ad->pipeline);
  ad->pipeline = NULL;

  return status;
}

int data_run_inference(appdata_s *ad) {
  char jpeg_path[PATH_MAX];

  int status = APP_ERROR_NONE;

  status = util_get_data_path("test.jpg", jpeg_path);
  if (status != APP_ERROR_NONE) {
    LOG_E("getting data path failed");
    return status;
  }

  status = util_save_drawing(ad->cr_surface, jpeg_path);
  if (status != APP_ERROR_NONE) {
    LOG_E("saving the cairo drawing failed");
    return status;
  }

  status = run_inference_pipeline_(ad, jpeg_path);

  return status;
}
