// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file main.c
 * @date 14 May 2020
 * @brief TIZEN Native Example App main entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <pthread.h>

#include <tizen.h>

#include "data.h"
#include "main.h"
#include "view.h"

static int routes_to_(appdata_s *ad, const char *source) {
  int status = view_routes_to(ad, source);
  if (status != 0) {
    LOG_E("routing to a new view failed for %s", source);
  }

  elm_layout_signal_callback_add(ad->layout, "routes/to", "*",
                                 &presenter_on_routes_request, ad);

  return status;
}

/**
 * @brief main thread runner wrapper for adding back callback again
 *
 * @param data
 */
static void notify_train_done(void *data) {
  appdata_s *ad = (appdata_s *)data;

  char buf[256];

  const char *source = "train_result";
  /// Throttle the function to slow down to incorporate with user interaction
  sleep(1);

  eext_object_event_callback_add(ad->naviframe, EEXT_CALLBACK_BACK,
                                 presenter_on_back_button_press, ad);

  int status = view_routes_to(ad, source);
  if (status != 0) {
    LOG_E("routing to a new view failed for %s", source);
    return;
  }

  elm_layout_signal_callback_add(ad->layout, "to_main", "",
                                 &presenter_on_go_main_request, ad);

  snprintf(buf, 255, "acc: %.0f%%", ad->best_accuracy);
  elm_object_part_text_set(ad->layout, "train_result/go_back/label", buf);
}

static void *train_(void *data) {
  int status = ML_ERROR_NONE;
  appdata_s *ad = (appdata_s *)data;
  status = pipe(ad->pipe_fd);
  if (status < 0) {
    LOG_E("opening pipe for training failed");
    goto RESTORE_CB;
  }

  ad->best_accuracy = 0.0;

  LOG_D("creating thread to run model");
  status = pthread_create(&ad->tid_writer, NULL, data_run_model, (void *)ad);
  if (status < 0) {
    LOG_E("creating pthread failed %s", strerror(errno));
    goto RESTORE_CB;
  }

  status =
    pthread_create(&ad->tid_reader, NULL, data_update_train_result, (void *)ad);
  if (status < 0) {
    LOG_E("creating pthread failed %s", strerror(errno));
    pthread_cancel(ad->tid_writer);
    goto RESTORE_CB;
  }

  status = pthread_join(ad->tid_reader, NULL);
  if (status < 0) {
    LOG_E("joining reader thread failed %s", strerror(errno));
    pthread_cancel(ad->tid_reader);
  }

  status = pthread_join(ad->tid_writer, NULL);
  if (status < 0) {
    LOG_E("joining writing thread failed %s", strerror(errno));
    pthread_cancel(ad->tid_writer);
  }

RESTORE_CB:
  ecore_main_loop_thread_safe_call_async(&notify_train_done, data);

  return NULL;
}

static int init_page_(appdata_s *ad, const char *path) {
  int status = APP_ERROR_NONE;

  if (!strcmp(path, "draw")) {
    ad->tries = 0;

    status = view_init_canvas(ad);
    if (status != APP_ERROR_NONE) {
      LOG_E("initiating canvas failed");
      return status;
    }

    view_set_canvas_clean(ad);

    if (ad->draw_target == INFER) {
      elm_layout_signal_callback_add(ad->layout, "draw/proceed", "",
                                     presenter_on_canvas_submit_inference, ad);
    } else if (ad->draw_target == TRAIN_UNSET) {
      elm_layout_signal_callback_add(ad->layout, "draw/proceed", "",
                                     presenter_on_canvas_submit_training, ad);
    } else {
      LOG_E("undefined draw target in initiation");
      return APP_ERROR_INVALID_CONTEXT;
    }
    return status;
  }

  return status;
}

void presenter_on_back_button_press(void *data, Evas_Object *obj,
                                    void *event_info EINA_UNUSED) {
  appdata_s *ad = data;

  ad->tries = 0;
  view_pop_naviframe(ad);
}

void presenter_on_routes_request(void *data, Evas_Object *obj EINA_UNUSED,
                                 const char *emission EINA_UNUSED,
                                 const char *source) {
  char *path, *path_data;
  appdata_s *ad = (appdata_s *)data;

  int status = util_parse_route(source, &path, &path_data);
  if (status) {
    LOG_E("something wrong with parsing %s", source);
    return;
  }

  LOG_D("%s %s", path, path_data);
  if (routes_to_(ad, path) != 0)
    return;

  /// check if path and path_data should be handled in special way,
  data_handle_path_data(ad, path_data);
  init_page_(ad, path);
}

void presenter_on_go_main_request(void *data, Evas_Object *obj EINA_UNUSED,
                                  const char *emission EINA_UNUSED,
                                  const char *source) {
  appdata_s *ad = (appdata_s *)data;
  elm_naviframe_item_pop_to(ad->home);
}

void presenter_on_canvas_submit_inference(void *data, Evas_Object *obj,
                                          const char *emission,
                                          const char *source) {
  appdata_s *ad = (appdata_s *)data;
  /** appdata handling NYI */
  data_run_inference(ad);

  ad->tries = 0;
  elm_naviframe_item_pop(ad->naviframe);
  if (routes_to_(ad, "test_result") != 0)
    return;
}

void presenter_on_canvas_submit_training(void *data, Evas_Object *obj,
                                         const char *emission,
                                         const char *source) {
  appdata_s *ad = (appdata_s *)data;
  int status = APP_ERROR_NONE;

  status = data_update_draw_target(ad);
  if (status != APP_ERROR_NONE) {
    LOG_E("setting draw target failed");
    return;
  }

  status = data_extract_feature(ad);
  if (status != APP_ERROR_NONE) {
    LOG_E("feature extraction failed");
    return;
  }

  if (ad->tries == MAX_TRIES - 1) {
    ad->tries = 0;
    elm_naviframe_item_pop(ad->naviframe);
    routes_to_((appdata_s *)data, "train_progress");
    pthread_t train_thread;
    eext_object_event_callback_del(ad->naviframe, EEXT_CALLBACK_BACK,
                                   presenter_on_back_button_press);
    status = pthread_create(&train_thread, NULL, train_, (void *)ad);
    if (status < 0) {
      LOG_E("creating pthread failed %s", strerror(errno));
      return;
    }
    status = pthread_detach(train_thread);
    if (status < 0) {
      LOG_E("detaching reading thread failed %s", strerror(errno));
      pthread_cancel(train_thread);
    }
  }

  /// prepare next canvas
  ad->tries++;
  view_set_canvas_clean(ad);
}

/********************* app related methods  **************************/
static bool app_create(void *data) {
  /* Hook to take necessary actions before main event loop starts
     Initialize UI resources and application's data
     If this function returns true, the main loop of application starts
     If this function returns false, the application is terminated */
  appdata_s *ad = data;
  char *data_path = app_get_data_path();

  pthread_mutex_init(&ad->pipe_lock, NULL);
  pthread_cond_init(&ad->pipe_cond, NULL);
  ad->data_output_pipe = ecore_pipe_add(view_update_result_cb, (void *)ad);
  if (ad->data_output_pipe == NULL) {
    LOG_E("making data out pipe failed");
    free(data_path);
    return false;
  }

  util_get_resource_path(EDJ_PATH, ad->edj_path, false);

  if (chdir(data_path) < 0) {
    LOG_E("change root directory failed");
    free(data_path);
    return false;
  };
  free(data_path);

  view_init(ad);
  eext_object_event_callback_add(ad->naviframe, EEXT_CALLBACK_BACK,
                                 presenter_on_back_button_press, ad);
  eext_object_event_callback_add(ad->naviframe, EEXT_CALLBACK_MORE,
                                 eext_naviframe_more_cb, NULL);

  if (routes_to_(ad, "home"))
    return false;

  ad->home = ad->nf_it;
  return true;
}

static void app_control(app_control_h app_control, void *data) {
  /* Handle the launch request. */
}

static void app_pause(void *data) {
  /* Take necessa
  ry actions when application becomes invisible. */
}

static void app_resume(void *data) {
  /* Take necessary actions when application becomes visible. */
}

static void app_terminate(void *data) { /* Release all resources. */
  appdata_s *ad = data;

  pthread_mutex_destroy(&ad->pipe_lock);
  pthread_cond_destroy(&ad->pipe_cond);
  ecore_pipe_del(ad->data_output_pipe);
}

static void ui_app_lang_changed(app_event_info_h event_info, void *user_data) {
  /*APP_EVENT_LANGUAGE_CHANGED*/

  int ret;
  char *language;

  ret = app_event_get_language(event_info, &language);
  if (ret != APP_ERROR_NONE) {
    LOG_E("app_event_get_language() failed. Err = %d.", ret);
    return;
  }

  if (language != NULL) {
    elm_language_set(language);
    free(language);
  }
}

static void ui_app_orient_changed(app_event_info_h event_info,
                                  void *user_data) {
  /*APP_EVENT_DEVICE_ORIENTATION_CHANGED*/
  return;
}

static void ui_app_region_changed(app_event_info_h event_info,
                                  void *user_data) {
  /*APP_EVENT_REGION_FORMAT_CHANGED*/
}

static void ui_app_low_battery(app_event_info_h event_info, void *user_data) {
  /*APP_EVENT_LOW_BATTERY*/
}

static void ui_app_low_memory(app_event_info_h event_info, void *user_data) {
  /*APP_EVENT_LOW_MEMORY*/
}

int main(int argc, char *argv[]) {
  appdata_s ad = {
    0,
  };
  int ret = 0;

  ui_app_lifecycle_callback_s event_callback = {
    0,
  };
  app_event_handler_h handlers[5] = {
    NULL,
  };

  event_callback.create = app_create;
  event_callback.terminate = app_terminate;
  event_callback.pause = app_pause;
  event_callback.resume = app_resume;
  event_callback.app_control = app_control;

  ui_app_add_event_handler(&handlers[APP_EVENT_LOW_BATTERY],
                           APP_EVENT_LOW_BATTERY, ui_app_low_battery, &ad);
  ui_app_add_event_handler(&handlers[APP_EVENT_LOW_MEMORY],
                           APP_EVENT_LOW_MEMORY, ui_app_low_memory, &ad);
  ui_app_add_event_handler(&handlers[APP_EVENT_DEVICE_ORIENTATION_CHANGED],
                           APP_EVENT_DEVICE_ORIENTATION_CHANGED,
                           ui_app_orient_changed, &ad);
  ui_app_add_event_handler(&handlers[APP_EVENT_LANGUAGE_CHANGED],
                           APP_EVENT_LANGUAGE_CHANGED, ui_app_lang_changed,
                           &ad);
  ui_app_add_event_handler(&handlers[APP_EVENT_REGION_FORMAT_CHANGED],
                           APP_EVENT_REGION_FORMAT_CHANGED,
                           ui_app_region_changed, &ad);

  ret = ui_app_main(argc, argv, &event_callback, &ad);
  if (ret != APP_ERROR_NONE) {
    dlog_print(DLOG_ERROR, LOG_TAG, "ui_app_main() is failed. err = %d", ret);
  }

  return ret;
}
