/**
 * @file view.c
 * @date 15 May 2020
 * @brief TIZEN Native Example App view entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include "view.h"

static Evas_Object *_create_layout(Evas_Object *parent, const char *edj_path,
                                   const char *group_name,
                                   Eext_Event_Cb back_cb, void *user_data);

static void _on_win_delete(void *data, Evas_Object *obj, void *event_info) {
  ui_app_exit();
}

static void _on_routes_to(void *data, Evas_Object *obj, const char *emission,
                          const char *source);

/**
 * @brief initiate window and conformant.
 * @param[in] ad appdata of the app
 * @retval #APP_ERROR_*
 */
int view_init(appdata_s *ad) {
  int status = APP_ERROR_NONE;

  Evas_Object *win, *conform, *nf;
  win = elm_win_util_standard_add(PACKAGE, PACKAGE);

  if (win == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to get create window err = %d",
               status);
    return status;
  }
  elm_win_conformant_set(win, EINA_TRUE);
  elm_win_autodel_set(win, EINA_TRUE);

  if (elm_win_wm_rotation_supported_get(win)) {
    int rots[4] = {0, 90, 180, 270};
    elm_win_wm_rotation_available_rotations_set(win, (const int *)(&rots), 4);
  }

  evas_object_smart_callback_add(win, "delete,request", _on_win_delete, NULL);
  evas_object_show(win);

  // Adding conformant
  conform = elm_conformant_add(win);
  if (conform == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to create conformant object");
    evas_object_del(win);
    return APP_ERROR_INVALID_CONTEXT;
  }
  elm_win_indicator_mode_set(win, ELM_WIN_INDICATOR_SHOW);
  elm_win_indicator_opacity_set(win, ELM_WIN_INDICATOR_OPAQUE);
  evas_object_size_hint_weight_set(conform, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
  elm_win_resize_object_add(win, conform);
  evas_object_show(conform);

  // Adding naviframe
  nf = elm_naviframe_add(conform);
  if (nf == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to create naviframe object");
    evas_object_del(win);
    return APP_ERROR_INVALID_CONTEXT;
  }

  elm_object_part_content_set(conform, "elm.swallow.content", nf);
  eext_object_event_callback_add(nf, EEXT_CALLBACK_BACK, eext_naviframe_back_cb,
                                 NULL);
  eext_object_event_callback_add(nf, EEXT_CALLBACK_MORE, eext_naviframe_more_cb,
                                 NULL);

  evas_object_show(nf);

  ad->circle_nf = eext_circle_surface_naviframe_add(nf);
  ad->naviframe = nf;
  ad->win = win;
  ad->conform = conform;

  return status;
}

/**
 * @brief creates layout from edj
 * @param[in] ad app data of the add
 * @param[in] group_name name of the layout to be pushed to main naviframe.
 */
int view_routes_to(appdata_s *ad, const char *group_name) {
  ad->layout =
    _create_layout(ad->naviframe, ad->edj_path, group_name, NULL, NULL);

  if (ad->layout == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to create a layout of no alarm.");
    evas_object_del(ad->win);
    return APP_ERROR_INVALID_CONTEXT;
  }

  elm_layout_signal_callback_add(ad->layout, "routes/to", "*", _on_routes_to,
                                 ad);

  if (!elm_naviframe_item_push(ad->naviframe, NULL, NULL, NULL, ad->layout,
                               "empty"))
    return APP_ERROR_INVALID_PARAMETER;

  return APP_ERROR_NONE;
}

static void _on_routes_to(void *data, Evas_Object *obj, const char *emission,
                          const char *source) {
  view_routes_to((appdata_s *)data, source);
}
/**
 * @brief creates a layout for parent object with EDJ file
 * @param[in] parent Parent object to attach to
 * @param[in] file_path EDJ file path
 * @param[in] group_name group name from edj
 * @param[in] back_cb callback when back even fired.
 * @param[in] user_data data to pass to the callback
 */
static Evas_Object *_create_layout(Evas_Object *parent, const char *edj_path,
                                   const char *group_name,
                                   Eext_Event_Cb back_cb, void *user_data) {
  Evas_Object *layout = NULL;

  if (parent == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "parent cannot be NULL");
    return NULL;
  }

  layout = elm_layout_add(parent);
  elm_layout_file_set(layout, edj_path, group_name);

  if (layout == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "There was error making layout");
    evas_object_del(layout);
    return NULL;
  }

  if (back_cb)
    eext_object_event_callback_add(layout, EEXT_CALLBACK_BACK, back_cb,
                                   user_data);

  evas_object_show(layout);

  return layout;
}
