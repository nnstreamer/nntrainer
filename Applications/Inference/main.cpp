// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_nntrainer_tensorfilter.cpp
 * @date 11 Sep 2020
 * @brief NNTrainer-NNStreamer tensorfilter tester.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <iostream>

#include <glib.h>
#include <gst/gst.h>

typedef struct {
  GMainLoop *loop;      /**< main event loop */
  GstElement *pipeline; /**< gst pipeline for data stream */
  GstBus *bus;          /**< gst bus for data pipeline */
  GstMessage *msg;
} AppData;

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  const gchar config_path[] = "./mnist.ini";
  const gchar input_path[] = "./img.png";

  AppData ad;

  gst_init(&argc, &argv);

  ad.loop = g_main_loop_new(NULL, FALSE);
  if (ad.loop == NULL) {
    std::cerr << "error making loop" << std::endl;
    return 1;
  }

  gchar *str_pipeline = g_strdup_printf(
    "filesrc location=%s ! pngdec ! tensor_converter ! "
    "tensor_transform mode=typecast option=float32 ! tensor_filter "
    "framework=nntrainer model=%s input=1:28:28:1 "
    "inputtype=float32 output=1:10:1:1 outputtype=float32 ! filesink "
    "location=nntrainer.out.1.log",
    input_path, config_path);

  ad.pipeline = gst_parse_launch(str_pipeline, NULL);
  g_free(str_pipeline);
  if (ad.pipeline == NULL) {
    g_main_loop_unref(ad.loop);
    ad.loop = NULL;
    std::cerr << "error launch pipeline" << std::endl;
    return 2;
  }

  gst_element_set_state(ad.pipeline, GST_STATE_PLAYING);

  ad.bus = gst_element_get_bus(ad.pipeline);
  if (ad.bus == NULL) {
    g_main_loop_unref(ad.loop);
    ad.loop = NULL;

    gst_object_unref(ad.pipeline);
    ad.pipeline = NULL;

    std::cerr << "error launch bus" << std::endl;
    return 3;
  }
  ad.msg = gst_bus_timed_pop_filtered(
    ad.bus, GST_CLOCK_TIME_NONE,
    (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

  if (ad.msg == NULL) {
    g_main_loop_unref(ad.loop);
    ad.loop = NULL;

    gst_object_unref(ad.pipeline);
    ad.pipeline = NULL;

    gst_object_unref(ad.bus);
    ad.bus = NULL;

    std::cerr << "error getting msg" << std::endl;
    return 4;
  }

  gst_message_unref(ad.msg);
  ad.msg = NULL;

  gst_object_unref(ad.bus);
  ad.bus = NULL;

  gst_element_set_state(ad.pipeline, GST_STATE_NULL);
  gst_object_unref(ad.pipeline);
  ad.pipeline = NULL;

  return 0;
}
