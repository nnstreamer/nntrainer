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

static int widget_instance_create(widget_context_h context, bundle *content,
                                  int w, int h, void *user_data) {
  int status = WIDGET_ERROR_NONE;
  widget_instance_data_s *wid =
    (widget_instance_data_s *)malloc(sizeof(widget_instance_data_s));

  if (wid == NULL) {
    status = WIDGET_ERROR_FAULT;
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to get window. err = %d", status);
    return status;
  }

  if (content != NULL) {
    /* Recover the previous status with the bundle object. */
  }
  widget_app_context_set_tag(context, wid);

  /* Window */
  status = view_create(context, w, h);
  if (status != WIDGET_ERROR_NONE) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to get window. err = %d", status);
    return WIDGET_ERROR_FAULT;
  }

  /* Label*/
  wid->label = elm_label_add(wid->conform);
  evas_object_resize(wid->label, w, h / 3);
  evas_object_move(wid->label, w / 4, h / 3);
  evas_object_show(wid->label);
  elm_object_text_set(wid->label, "Hello widget!");

  return WIDGET_ERROR_NONE;
}

static int widget_instance_destroy(widget_context_h context,
                                   widget_app_destroy_type_e reason,
                                   bundle *content, void *user_data) {

  int status = WIDGET_ERROR_NONE;
  widget_instance_data_s *wid = NULL;

  status = widget_app_context_get_tag(context, (void **)&wid);
  if (status != WIDGET_ERROR_NONE) {
    dlog_print(DLOG_ERROR, LOG_TAG, "failed to destroy. err = %d", status);
    return status;
  }

  if (wid->win)
    evas_object_del(wid->win);

  free(wid);

  return WIDGET_ERROR_NONE;
}

static int widget_instance_pause(widget_context_h context, void *user_data) {
  /* Take necessary actions when widget instance becomes invisible. */
  return WIDGET_ERROR_NONE;
}

static int widget_instance_resume(widget_context_h context, void *user_data) {
  /* Take necessary actions when widget instance becomes visible. */
  return WIDGET_ERROR_NONE;
}

static int widget_instance_update(widget_context_h context, bundle *content,
                                  int force, void *user_data) {
  /* Take necessary actions when widget instance should be updated. */
  return WIDGET_ERROR_NONE;
}

static int widget_instance_resize(widget_context_h context, int w, int h,
                                  void *user_data) {
  /* Take necessary actions when the size of widget instance was changed. */
  return WIDGET_ERROR_NONE;
}

static void widget_app_lang_changed(app_event_info_h event_info,
                                    void *user_data) {
  /* APP_EVENT_LANGUAGE_CHANGED */
  char *locale = NULL;
  app_event_get_language(event_info, &locale);
  elm_language_set(locale);
  free(locale);
}

static void widget_app_region_changed(app_event_info_h event_info,
                                      void *user_data) {
  /* APP_EVENT_REGION_FORMAT_CHANGED */
}

static widget_class_h widget_app_create(void *user_data) {
  /* Hook to take necessary actions before main event loop starts.
     Initialize UI resources.
     Make a class for widget instance.
  */
  app_event_handler_h handlers[5] = {
    NULL,
  };

  widget_app_add_event_handler(&handlers[APP_EVENT_LANGUAGE_CHANGED],
                               APP_EVENT_LANGUAGE_CHANGED,
                               widget_app_lang_changed, user_data);
  widget_app_add_event_handler(&handlers[APP_EVENT_REGION_FORMAT_CHANGED],
                               APP_EVENT_REGION_FORMAT_CHANGED,
                               widget_app_region_changed, user_data);

  widget_instance_lifecycle_callback_s ops = {
    .create = widget_instance_create,
    .destroy = widget_instance_destroy,
    .pause = widget_instance_pause,
    .resume = widget_instance_resume,
    .update = widget_instance_update,
    .resize = widget_instance_resize,
  };

  return widget_app_class_create(ops, user_data);
}

static void widget_app_terminate(void *user_data) {
  /* Release all resources. */
}

int main(int argc, char *argv[]) {
  widget_app_lifecycle_callback_s ops = {
    0,
  };
  int ret;

  ops.create = widget_app_create;
  ops.terminate = widget_app_terminate;

  ret = widget_app_main(argc, argv, &ops, NULL);
  if (ret != WIDGET_ERROR_NONE) {
    dlog_print(DLOG_ERROR, LOG_TAG, "widget_app_main() is failed. err = %d",
               ret);
  }

  return ret;
}
