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

/**
 * @brief initiate window and conformant.
 * @param[in] context context of widget
 * @param[in] w width
 * @param[in] h height
 * @retval #WIDGET_ERROR_*
 */
int view_init(widget_context_h context, int w, int h) {
  widget_instance_data_s *wid = NULL;
  int status = WIDGET_ERROR_NONE;

  Evas_Object *win, *conform, *nf;

  status = widget_app_get_elm_win(context, &win);

  if (status != WIDGET_ERROR_NONE) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to get create window err = %d",
               status);
    return status;
  }
  evas_object_resize(win, w, h);
  evas_object_show(win);

  // Adding conformant
  conform = elm_conformant_add(win);
  if (conform == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to create conformant object");
    evas_object_del(win);
    return WIDGET_ERROR_FAULT;
  }

  evas_object_size_hint_weight_set(conform, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
  elm_win_resize_object_add(win, conform);
  evas_object_show(conform);

  // Adding naviframe
  nf = elm_naviframe_add(conform);
  if (nf == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to create naviframe object");
    evas_object_del(win);
    return WIDGET_ERROR_FAULT;
  }

  elm_object_part_content_set(conform, "elm.swallow.content", nf);
  eext_object_event_callback_add(nf, EEXT_CALLBACK_BACK, eext_naviframe_back_cb,
                                 NULL);
  eext_object_event_callback_add(nf, EEXT_CALLBACK_MORE, eext_naviframe_more_cb,
                                 NULL);

  evas_object_show(nf);

  widget_app_context_get_tag(context, (void **)&wid);

  wid->circle_nf = eext_circle_surface_naviframe_add(nf);

  wid->naviframe = nf;
  wid->win = win;
  wid->conform = conform;

  return status;
}

/**
 * @brief creates layout from edj
 * @param[in] context context of widget instance
 */
int view_create(widget_context_h context) {
  widget_instance_data_s *wid = NULL;
  int status = WIDGET_ERROR_NONE;

  widget_app_context_get_tag(context, (void **)&wid);
  wid->layout =
    _create_layout(wid->naviframe, wid->edj_path, "home", NULL, NULL);

  if (wid->layout == NULL) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to create a layout of no alarm.");
    evas_object_del(wid->win);
    return WIDGET_ERROR_FAULT;
  }

  elm_naviframe_item_push(wid->naviframe, NULL, NULL, NULL, wid->layout,
                          "empty");
  return status;
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
