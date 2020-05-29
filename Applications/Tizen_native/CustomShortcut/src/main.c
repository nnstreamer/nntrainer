/**
 * @file main.c
 * @date 14 May 2020
 * @brief TIZEN Native Example App main entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include "main.h"
#include "view.h"
#include <tizen.h>

static void win_delete_request_cb(void *data, Evas_Object *obj,
                                  void *event_info) {
  ui_app_exit();
}

static void layout_back_cb(void *data, Evas_Object *obj, void *event_info) {
  appdata_s *ad = data;
  /* Let window go to hide state. */
  elm_win_lower(ad->win);
}

static void app_get_resource(const char *edj_file_in, char *edj_path_out,
                             int edj_path_max) {
  char *res_path = app_get_resource_path();
  if (res_path) {
    snprintf(edj_path_out, edj_path_max, "%s%s", res_path, edj_file_in);
    free(res_path);
  }
}

static void create_base_gui(appdata_s *ad) {
  char edj_path[PATH_MAX] = {
    0,
  };

  /* Window */
  /* Create and initialize elm_win.
     elm_win is mandatory to manipulate window. */
  ad->win = elm_win_util_standard_add(PACKAGE, PACKAGE);
  elm_win_conformant_set(ad->win, EINA_TRUE);
  elm_win_autodel_set(ad->win, EINA_TRUE);

  if (elm_win_wm_rotation_supported_get(ad->win)) {
    int rots[4] = {0, 90, 180, 270};
    elm_win_wm_rotation_available_rotations_set(ad->win, (const int *)(&rots),
                                                4);
  }

  evas_object_smart_callback_add(ad->win, "delete,request",
                                 win_delete_request_cb, NULL);

  /* Conformant */
  /* Create and initialize elm_conformant.
     elm_conformant is mandatory for base gui to have proper size
     when indicator or virtual keypad is visible. */
  ad->conform = elm_conformant_add(ad->win);
  elm_win_indicator_mode_set(ad->win, ELM_WIN_INDICATOR_SHOW);
  elm_win_indicator_opacity_set(ad->win, ELM_WIN_INDICATOR_OPAQUE);
  evas_object_size_hint_weight_set(ad->conform, EVAS_HINT_EXPAND,
                                   EVAS_HINT_EXPAND);
  elm_win_resize_object_add(ad->win, ad->conform);
  evas_object_show(ad->conform);

  /* Base Layout */
  /* Create an actual view of the base gui.
     Modify this part to change the view. */
  app_get_resource(EDJ_PATH, edj_path, (int)PATH_MAX);
  ad->layout = elm_layout_add(ad->win);
  elm_layout_file_set(ad->layout, edj_path, "home");
  evas_object_size_hint_weight_set(ad->layout, EVAS_HINT_EXPAND,
                                   EVAS_HINT_EXPAND);
  eext_object_event_callback_add(ad->layout, EEXT_CALLBACK_BACK, layout_back_cb,
                                 ad);
  elm_object_content_set(ad->conform, ad->layout);

  /* Show window after base gui is set up */
  evas_object_show(ad->win);
}

static bool app_create(void *data) {
  /* Hook to take necessary actions before main event loop starts
     Initialize UI resources and application's data
     If this function returns true, the main loop of application starts
     If this function returns false, the application is terminated */
  appdata_s *ad = data;
  char edj_path[PATH_MAX] = {
    0,
  };

  view_init(ad);

  view_routes_to(ad, "home");

  char *res_path = app_get_resource_path();
  if (res_path == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to get resource.");
    return false;
  }

  snprintf(ad->edj_path, sizeof(ad->edj_path), "%s%s", res_path, EDJ_PATH);
  free(res_path);

  if (view_routes_to(ad, "home"))
    return false;

  return true;
}

static void app_control(app_control_h app_control, void *data) {
  /* Handle the launch request. */
}

static void app_pause(void *data) {
  /* Take necessary actions when application becomes invisible. */
}

static void app_resume(void *data) {
  /* Take necessary actions when application becomes visible. */
}

static void app_terminate(void *data) { /* Release all resources. */ }

static void ui_app_lang_changed(app_event_info_h event_info, void *user_data) {
  /*APP_EVENT_LANGUAGE_CHANGED*/

  int ret;
  char *language;

  ret = app_event_get_language(event_info, &language);
  if (ret != APP_ERROR_NONE) {
    dlog_print(DLOG_ERROR, LOG_TAG,
               "app_event_get_language() failed. Err = %d.", ret);
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
