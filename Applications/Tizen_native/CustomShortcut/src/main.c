// SPDX-License-Identifier: Apache-2.0-only
/**
 * @file main.c
 * @date 14 May 2020
 * @brief TIZEN Native Example App main entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include "main.h"
#include "data.h"
#include "view.h"
#include <pthread.h>
#include <tizen.h>

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

  data_get_resource_path(EDJ_PATH, ad->edj_path, false);

  if (chdir(data_path) < 0) {
    LOG_E("change root directory failed");
    free(data_path);
    return false;
  };
  free(data_path);

  view_init(ad);

  if (view_routes_to(ad, "home"))
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
