/**
 * @file view.c
 * @date 15 May 2020
 * @brief TIZEN Native Example App view entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include "view.h"

/**
 * @brief create view
 */
int view_create(widget_context_h context, int w, int h) {
  widget_instance_data_s *wid = NULL;
  int status = WIDGET_ERROR_NONE;

  Evas_Object *win, *conform;

  status = widget_app_get_elm_win(context, &win);

  if (status != WIDGET_ERROR_NONE) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to get create window err = %d",
               status);
    return status;
  }
  evas_object_resize(win, w, h);
  evas_object_show(win);

  conform = elm_conformant_add(win);

  evas_object_size_hint_weight_set(conform, EVAS_HINT_EXPAND, EVAS_HINT_EXPAND);
  elm_win_resize_object_add(win, conform);
  evas_object_show(conform);

  widget_app_context_get_tag(context, (void **)&wid);

  wid->win = win;
  wid->conform = conform;

  return status;
}
