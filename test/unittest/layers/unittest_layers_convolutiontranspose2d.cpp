// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_convolution.cpp
 * @date 5 July 2021
 * @brief ConvTranspose2d Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <convtranspose2d_layer.h>
#include <layers_common_tests.h>

auto semantic_convtranspose2d = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>, nntrainer::ConvTranspose2DLayer::type,
  {"filters=1", "kernel_size=1,1", "padding=1,1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(ConvolutionTranspose2D, LayerSemantics,
                     ::testing::Values(semantic_convtranspose2d));

auto convtranspose2d_sb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "1:1:4:4",
  "convtranspose2d_sb_minimum.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_mb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "3:1:4:4",
  "convtranspose2d_mb_minimum.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_sb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "1:1:4:4",
  "convtranspose2d_sb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_mb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "3:1:4:4",
  "convtranspose2d_mb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_sb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "1:3:4:4", "convtranspose2d_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_sb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "1:3:4:4", "convtranspose2d_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_mb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "3:3:4:4", "convtranspose2d_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_mb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "3:3:4:4", "convtranspose2d_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_sb_valid_drop_last =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
                           {
                             "filters=2",
                             "kernel_size=3,3",
                             "stride=2,2",
                             "padding=valid",
                           },
                           "1:3:7:7", "convtranspose2d_sb_valid_drop_last.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_mb_valid_drop_last =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
                           {
                             "filters=2",
                             "kernel_size=3,3",
                             "stride=2,2",
                             "padding=valid",
                           },
                           "3:3:7:7", "convtranspose2d_mb_valid_drop_last.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_sb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {"filters=3", "kernel_size=2,2", "stride=3,3"}, "1:2:5:5",
  "convtranspose2d_sb_no_overlap.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_mb_no_overlap =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
                           {
                             "filters=3",
                             "kernel_size=2,2",
                             "stride=3,3",
                           },
                           "3:2:5:5", "convtranspose2d_mb_no_overlap.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_sb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
  {"filters=3", "kernel_size=1,1", "stride=2,2"}, "1:2:5:5",
  "convtranspose2d_sb_1x1_kernel.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto convtranspose2d_mb_1x1_kernel =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
                           {
                             "filters=3",
                             "kernel_size=1,1",
                             "stride=2,2",
                           },
                           "3:2:5:5", "convtranspose2d_mb_1x1_kernel.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

// auto convtranspose2d_sb_dilation =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
//                            {
//                              "filters=2",
//                              "kernel_size=3,3",
//                              "dilation=2,2",
//                            },
//                            "1:3:11:11", "convtranspose2d_sb_dilation.nnlayergolden",
//                            LayerGoldenTestParamOptions::DEFAULT);

// auto convtranspose2d_mb_dilation =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
//                            {
//                              "filters=2",
//                              "kernel_size=3,3",
//                              "dilation=2,2",
//                            },
//                            "3:3:11:11", "convtranspose2d_mb_dilation.nnlayergolden",
//                            LayerGoldenTestParamOptions::DEFAULT);

// auto convtranspose2d_sb_same_dilation =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
//                            {
//                              "filters=2",
//                              "kernel_size=3,3",
//                              "padding=same",
//                              "dilation=2,2",
//                            },
//                            "1:3:11:11", "convtranspose2d_sb_same_dilation.nnlayergolden",
//                            LayerGoldenTestParamOptions::DEFAULT);

// auto convtranspose2d_mb_same_dilation =
//   LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::ConvTranspose2DLayer>,
//                            {
//                              "filters=2",
//                              "kernel_size=3,3",
//                              "padding=same",
//                              "dilation=2,2",
//                            },
//                            "3:3:11:11", "convtranspose2d_mb_same_dilation.nnlayergolden",
//                            LayerGoldenTestParamOptions::DEFAULT);

GTEST_PARAMETER_TEST(
  ConvolutionTranspose2D, LayerGoldenTest,
  ::testing::Values(
    convtranspose2d_sb_minimum, convtranspose2d_mb_minimum, convtranspose2d_sb_same_remain,
    convtranspose2d_mb_same_remain, convtranspose2d_sb_same_uneven_remain_1,
    convtranspose2d_sb_same_uneven_remain_2, convtranspose2d_mb_same_uneven_remain_1,
    convtranspose2d_mb_same_uneven_remain_2, convtranspose2d_sb_valid_drop_last,
    convtranspose2d_mb_valid_drop_last, convtranspose2d_sb_no_overlap, convtranspose2d_mb_no_overlap,
    convtranspose2d_sb_1x1_kernel, convtranspose2d_mb_1x1_kernel
    // ,convtranspose2d_sb_dilation, convtranspose2d_mb_dilation, convtranspose2d_sb_same_dilation, convtranspose2d_mb_same_dilation
    ));
