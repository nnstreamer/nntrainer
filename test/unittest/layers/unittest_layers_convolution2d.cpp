// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_convolution.cpp
 * @date 5 July 2021
 * @brief Conv2d Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <conv2d_layer.h>
#include <layers_common_tests.h>

auto semantic_conv2d = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>, nntrainer::Conv2DLayer::type,
  {"filters=1", "kernel_size=1,1", "padding=1,1"}, 0, false, 1);

INSTANTIATE_TEST_CASE_P(Convolution2D, LayerSemantics,
                        ::testing::Values(semantic_conv2d));

auto conv_sb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "1:1:4:4", "conv_sb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto conv_mb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2"}, "3:1:4:4", "conv_mb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto conv_sb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "1:1:4:4",
  "conv_sb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto conv_mb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=2", "kernel_size=3,3", "padding=same"}, "3:1:4:4",
  "conv_mb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto conv_sb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "1:3:4:4", "conv_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto conv_sb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "1:3:4:4", "conv_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto conv_mb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=same",
  },
  "3:3:4:4", "conv_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto conv_mb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {
    "filters=2",
    "kernel_size=3,3",
    "stride=2,2",
    "padding=0,1,0,1",
  },
  "3:3:4:4", "conv_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto conv_sb_valid_drop_last =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::Conv2DLayer>,
                           {
                             "filters=2",
                             "kernel_size=3,3",
                             "stride=2,2",
                             "padding=valid",
                           },
                           "1:3:7:7", "conv_sb_valid_drop_last.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

auto conv_mb_valid_drop_last =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::Conv2DLayer>,
                           {
                             "filters=2",
                             "kernel_size=3,3",
                             "stride=2,2",
                             "padding=valid",
                           },
                           "3:3:7:7", "conv_mb_valid_drop_last.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

auto conv_sb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=2,2", "stride=3,3"}, "1:2:5:5",
  "conv_sb_no_overlap.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto conv_mb_no_overlap =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::Conv2DLayer>,
                           {
                             "filters=3",
                             "kernel_size=2,2",
                             "stride=3,3",
                           },
                           "3:2:5:5", "conv_mb_no_overlap.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

auto conv_sb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv2DLayer>,
  {"filters=3", "kernel_size=1,1", "stride=2,2"}, "1:2:5:5",
  "conv_sb_1x1_kernel.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto conv_mb_1x1_kernel =
  LayerGoldenTestParamType(nntrainer::createLayer<nntrainer::Conv2DLayer>,
                           {
                             "filters=3",
                             "kernel_size=1,1",
                             "stride=2,2",
                           },
                           "3:2:5:5", "conv_mb_1x1_kernel.nnlayergolden",
                           LayerGoldenTestParamOptions::DEFAULT);

INSTANTIATE_TEST_CASE_P(
  Convolution2D, LayerGoldenTest,
  ::testing::Values(conv_sb_minimum, conv_mb_minimum, conv_sb_same_remain,
                    conv_mb_same_remain, conv_sb_same_uneven_remain_1,
                    conv_sb_same_uneven_remain_2, conv_mb_same_uneven_remain_1,
                    conv_mb_same_uneven_remain_2, conv_sb_valid_drop_last,
                    conv_mb_valid_drop_last, conv_sb_no_overlap,
                    conv_mb_no_overlap, conv_sb_1x1_kernel,
                    conv_mb_1x1_kernel));
