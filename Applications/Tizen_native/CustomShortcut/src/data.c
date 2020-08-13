// SPDX-License-Identifier: Apache-2.0-only
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
#include "data.h"
#include <regex.h>
#include <stdio.h>
#include <string.h>

int data_parse_route(const char *source, char **route, char **data) {
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

int data_get_resource_path(const char *file, char *full_path, bool shared) {
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

int data_get_data_path(const char *file, char *full_path) {
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

static void _on_data_receive(ml_tensors_data_h data,
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

  bool target_label = ad->tries % 2;
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

static int _run_nnpipeline(appdata_s *ad, const char *src, bool append) {
  char pipe_description[5000];

  char model_path[PATH_MAX];

  int status = ML_ERROR_NONE;

  data_get_resource_path("mobilenetv2.tflite", model_path, false);

  status = pthread_mutex_lock(&ad->pipe_lock);
  if (status != 0) {
    LOG_E("acquiring lock failed status: %d", status);
    return status;
  }

  LOG_D("pipe ready, starting pipeline");

  sprintf(pipe_description,
          "filesrc location=%s ! pngdec ! "
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
    ml_pipeline_destroy(ad->pipeline);
    pthread_mutex_unlock(&ad->pipe_lock);
    return status;
  }

  status = ml_pipeline_sink_register(ad->pipeline, "sink", _on_data_receive,
                                     (void *)ad, &ad->pipe_sink);
  if (status != ML_ERROR_NONE) {
    LOG_E("sink register failed %d", status);
    goto CLEAN;
  }

  LOG_D("starting inference");
  status = ml_pipeline_start(ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("failed to start pipeline %d", status);
    goto CLEAN;
  }

  pthread_cond_wait(&ad->pipe_cond, &ad->pipe_lock);

  LOG_D("stopping pipeline");
  status = ml_pipeline_stop(ad->pipeline);
  if (status != ML_ERROR_NONE) {
    LOG_E("stopping pipeline failed");
    goto CLEAN;
  }

  LOG_D("unregister pipeline");
  if (status != ML_ERROR_NONE) {
    LOG_E("unregistering sink failed");
  }

CLEAN:
  LOG_D("destroying pipeline");
  ml_pipeline_sink_unregister(ad->pipe_sink);
  ml_pipeline_destroy(ad->pipeline);
  ad->pipe_sink = NULL;
  ad->pipeline = NULL;
  pthread_mutex_unlock(&ad->pipe_lock);
  return status;
}

int data_extract_feature(appdata_s *ad, const char *dst, bool append) {
  char png_path[PATH_MAX];
  cairo_status_t cr_stat = CAIRO_STATUS_SUCCESS;
  int status = APP_ERROR_NONE;

  data_get_data_path("temp.png", png_path);
  LOG_D("start writing to png_path: %s ", png_path);
  cr_stat = cairo_surface_write_to_png(ad->cr_surface, png_path);

  if (cr_stat != CAIRO_STATUS_SUCCESS) {
    LOG_E("failed to write cairo surface as a file reason: %d", cr_stat);
    return APP_ERROR_INVALID_PARAMETER;
  }

  data_get_data_path(dst, ad->pipe_dst);

  LOG_I("start inference to dataset: %s ", ad->pipe_dst);
  status = _run_nnpipeline(ad, png_path, append);

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

  printf("test");

  LOG_D("start running model");
  data_get_resource_path("model.ini", model_conf_path, false);
  data_get_data_path("label.dat", label_path);

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

int data_parse_result_string(const char *src, train_result_s *train_result) {
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
