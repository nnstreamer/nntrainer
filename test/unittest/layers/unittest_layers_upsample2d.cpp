// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 heka1024 <heka1024@gmail.com>
 *
 * @file unittest_layers_upsample2d.cppp
 * @date 15 June 2024
 * @brief Unit test for upsample2d layer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author heka1024 <heka1024@gmail.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <upsample2d_layer.h>

auto semantic_upsample2d = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Upsample2dLayer>,
  nntrainer::Upsample2dLayer::type, {"upsample=nearest", "kernel_size=2,2"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Upsample2D, LayerSemantics,
                     ::testing::Values(semantic_upsample2d));

auto upsampling_2x2_nearest = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Upsample2dLayer>,
  {"upsample=nearest", "kernel_size=2,2"}, "1:1:2:2",
  "upsample2d_2x2_nearest.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto upsampling_3x3_nearest = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Upsample2dLayer>,
  {"upsample=nearest", "kernel_size=3,3"}, "1:1:3:3",
  "upsample2d_3x3_nearest.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto upsampling_2x2_bilinear = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Upsample2dLayer>,
  {"upsample=bilinear", "kernel_size=2,2"}, "1:1:2:2",
  "upsample2d_2x2_bilinear.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto upsampling_3x3_bilinear = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Upsample2dLayer>,
  {"upsample=bilinear", "kernel_size=3,3"}, "1:1:3:3",
  "upsample2d_3x3_bilinear.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

auto upsampling_big_bilinear = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Upsample2dLayer>,
  {"upsample=bilinear", "kernel_size=4,4"}, "1:1:10:10",
  "upsample2d_big_bilinear.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(Upsample2D, LayerGoldenTest,
                     ::testing::Values(upsampling_2x2_nearest,
                                       upsampling_3x3_nearest,
                                       upsampling_2x2_bilinear,
                                       upsampling_3x3_bilinear,
                                       upsampling_big_bilinear));
