/**
 * @file view.c
 * @date 15 May 2020
 * @brief TIZEN Native Example App view entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include "view.h"
#include "data.h"

static Evas_Object *_create_layout(Evas_Object *parent, const char *edj_path,
                                   const char *group_name,
                                   Eext_Event_Cb back_cb, void *user_data);

static int _create_canvas(appdata_s *ad, const char *draw_mode);

static void _on_win_delete(void *data, Evas_Object *obj, void *event_info) {
  ui_app_exit();
}

static void _on_back_pressed(void *data, Evas_Object *obj, void *event_info) {
  appdata_s *ad = data;
  Elm_Widget_Item *nf_it = elm_naviframe_top_item_get(obj);

  if (!nf_it) {
    /* app should not reach hear */
    dlog_print(DLOG_ERROR, LOG_TAG, "naviframe is empty.");
    ui_app_exit();
    return;
  }

  if (nf_it == ad->home) {
    dlog_print(DLOG_DEBUG, LOG_TAG, "navi empty");
    elm_win_lower(ad->win);
    return;
  }

  dlog_print(DLOG_DEBUG, LOG_TAG, "item popped");
  elm_naviframe_item_pop(obj);
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
  eext_object_event_callback_add(nf, EEXT_CALLBACK_BACK, _on_back_pressed, ad);
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
 * @param[in/out] ad app data of the add
 * @param[in] group_name name of the layout to be pushed to main naviframe.
 */
int view_routes_to(appdata_s *ad, const char *group_name) {
  char *path, *path_data;
  int status;

  status = parse_route(group_name, &path, &path_data);
  if (status) {
    _E("something wrong with parsing %s", group_name);
    return status;
  }

  _D("%s %s", path, path_data);

  ad->layout = _create_layout(ad->naviframe, ad->edj_path, path, NULL, NULL);

  if (ad->layout == NULL) {
    _E("failed to create layout");
    status = APP_ERROR_INVALID_CONTEXT;
    evas_object_del(ad->win);
    goto CLEAN_UP;
  }

  ad->nf_it = elm_naviframe_item_push(ad->naviframe, NULL, NULL, NULL,
                                      ad->layout, "empty");

  if (ad->nf_it == NULL) {
    status = APP_ERROR_INVALID_PARAMETER;
    goto CLEAN_UP;
  }

  if (!strcmp(path, "draw")) {
    status = _create_canvas(ad, path_data);
  }

  elm_layout_signal_callback_add(ad->layout, "routes/to", "*", _on_routes_to,
                                 ad);

CLEAN_UP:
  free(path);
  return status;
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
 * @param[in] back_cb callback when back event fired.
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

static void _on_draw_start(void *data, Evas *e, Evas_Object *obj,
                           void *event_info) {
  appdata_s *ad = (appdata_s *)data;
  Evas_Event_Mouse_Down *eemd = (Evas_Event_Mouse_Down *)event_info;
  _D("x: %d, y: %d", eemd->canvas.x, eemd->canvas.y);

  cairo_set_source_rgba(ad->cr, 1, 1, 1, 1);
  cairo_move_to(ad->cr, eemd->canvas.x - ad->x_offset,
                eemd->canvas.y - ad->y_offset);
}

static void _on_draw_move(void *data, Evas *e, Evas_Object *obj,
                          void *event_info) {
  appdata_s *ad = (appdata_s *)data;
  Evas_Event_Mouse_Move *eemm = (Evas_Event_Mouse_Move *)event_info;

  _D("x: %d, y: %d", eemm->cur.canvas.x, eemm->cur.canvas.y);
  cairo_line_to(ad->cr, eemm->cur.canvas.x - ad->x_offset,
                eemm->cur.canvas.y - ad->y_offset);
}

static void _on_draw_end(void *data, Evas *e, Evas_Object *obj,
                         void *event_info) {
  appdata_s *ad = (appdata_s *)data;
  _D("draw end");
  cairo_stroke(ad->cr);
  cairo_surface_flush(ad->cr_surface);
  evas_object_image_data_update_add(ad->canvas, 0, 0, ad->width, ad->height);
}

static Eina_Bool _on_canvas_exit(void *data, Elm_Object_Item *it) {
  _D("exiting canvas");
  appdata_s *ad = (appdata_s *)data;
  /// @todo save file to loadable data format

  evas_object_del(ad->canvas);
  cairo_surface_destroy(ad->cr_surface);
  cairo_destroy(ad->cr);
  return EINA_TRUE;
}

static int _create_canvas(appdata_s *ad, const char *draw_mode) {
  _D("init canvas");
  Eina_Bool status;

  Evas_Object *frame = elm_layout_add(ad->layout);

  status = elm_layout_content_set(ad->layout, "draw/canvas", frame);
  if (status == EINA_FALSE) {
    _E("failed to get canvas object");
    return APP_ERROR_INVALID_PARAMETER;
  }

  evas_object_move(frame, 72, 72);
  evas_object_resize(frame, 216, 216);
  evas_object_show(frame);
  Evas_Coord width, height, x, y;

  evas_object_geometry_get(frame, &x, &y, &width, &height);
  _D("frame info, %d %d width: %d height: %d", x, y, width, height);

  Evas_Object *canvas =
    evas_object_image_filled_add(evas_object_evas_get(frame));
  if (canvas == NULL) {
    _E("failed to initiate canvas");
    return APP_ERROR_INVALID_PARAMETER;
  }

  evas_object_image_content_hint_set(canvas, EVAS_IMAGE_CONTENT_HINT_DYNAMIC);
  evas_object_image_size_set(canvas, width, height);
  evas_object_move(canvas, x, y);
  evas_object_resize(canvas, width, height);
  evas_object_image_colorspace_set(canvas, EVAS_COLORSPACE_ARGB8888);
  evas_object_image_alpha_set(canvas, 1);
  evas_object_show(canvas);

  ad->pixels = (unsigned char *)evas_object_image_data_get(canvas, 1);
  if (ad->pixels == NULL) {
    _E("cannot fetch pixels from image");
    return APP_ERROR_INVALID_PARAMETER;
  }

  cairo_surface_t *cairo_surface = cairo_image_surface_create_for_data(
    ad->pixels, CAIRO_FORMAT_ARGB32, width, height, width * 4);
  if (cairo_surface_status(cairo_surface) != CAIRO_STATUS_SUCCESS) {
    _E("cannot make cairo surface");
    evas_object_del(canvas);
    cairo_surface_destroy(cairo_surface);
    return APP_ERROR_INVALID_PARAMETER;
  }

  cairo_t *cr = cairo_create(cairo_surface);
  if (cairo_status(cr) != CAIRO_STATUS_SUCCESS) {
    _E("Cannot initiate cairo surface");
    evas_object_del(canvas);
    cairo_surface_destroy(cairo_surface);
    cairo_destroy(cr);
    return APP_ERROR_INVALID_PARAMETER;
  }

  cairo_rectangle(cr, 0, 0, width, height);
  cairo_set_source_rgba(cr, 0.5, 0.5, 0.5, 0.5);
  cairo_fill(cr);
  cairo_surface_flush(cairo_surface);

  evas_object_image_data_update_add(canvas, 0, 0, width, height);

  evas_object_event_callback_add(canvas, EVAS_CALLBACK_MOUSE_DOWN,
                                 _on_draw_start, (void *)ad);

  evas_object_event_callback_add(canvas, EVAS_CALLBACK_MOUSE_UP, _on_draw_end,
                                 (void *)ad);

  evas_object_event_callback_add(canvas, EVAS_CALLBACK_MOUSE_MOVE,
                                 _on_draw_move, (void *)ad);

  elm_naviframe_item_pop_cb_set(ad->nf_it, _on_canvas_exit, ad);

  ad->canvas = canvas;
  ad->cr_surface = cairo_surface;
  ad->cr = cr;
  ad->width = width;
  ad->height = height;
  ad->x_offset = x;
  ad->y_offset = y;

  return APP_ERROR_NONE;
}
