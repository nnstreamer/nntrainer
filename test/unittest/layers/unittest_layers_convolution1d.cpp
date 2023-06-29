// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_convolution.cpp
 * @date 18 October 2021
 * @brief Conv1d Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <conv1d_layer.h>
#include <layers_common_tests.h>

auto semantic_conv1d = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>, nntrainer::Conv1DLayer::type,
  {"filters=1", "kernel_size=1", "padding=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(Convolution1D, LayerSemantics,
                     ::testing::Values(semantic_conv1d));

auto conv1d_sb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=2"}, "1:1:1:4", "conv1d_sb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_mb_minimum = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=2"}, "3:1:1:4", "conv1d_mb_minimum.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=2", "kernel_size=3", "padding=same"}, "1:1:1:4",
  "conv1d_sb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto conv1d_mb_same_remain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=2", "kernel_size=3", "padding=same"}, "3:1:1:4",
  "conv1d_mb_same_remain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto conv1d_sb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "stride=2",
    "padding=same",
  },
  "1:3:1:4", "conv1d_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "stride=2",
    "padding=0,1",
  },
  "1:3:1:4", "conv1d_sb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_mb_same_uneven_remain_1 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "stride=2",
    "padding=same",
  },
  "3:3:1:4", "conv1d_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_mb_same_uneven_remain_2 = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "stride=2",
    "padding=0,1",
  },
  "3:3:1:4", "conv1d_mb_same_uneven_remain.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_valid_drop_last = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "stride=2",
    "padding=valid",
  },
  "1:3:1:7", "conv1d_sb_valid_drop_last.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_mb_valid_drop_last = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "stride=2",
    "padding=valid",
  },
  "3:3:1:7", "conv1d_mb_valid_drop_last.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=2", "stride=3"}, "1:2:1:5",
  "conv1d_sb_no_overlap.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto conv1d_mb_no_overlap = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=3",
    "kernel_size=2",
    "stride=3",
  },
  "3:2:1:5", "conv1d_mb_no_overlap.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_causal = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=2", "padding=causal"}, "1:1:1:4",
  "conv1d_sb_causal.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto conv1d_mb_causal = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=2", "padding=causal"}, "3:1:1:4",
  "conv1d_mb_causal.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto conv1d_sb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=1", "stride=2"}, "1:2:1:5",
  "conv1d_sb_1x1_kernel.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32");

auto conv1d_mb_1x1_kernel = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=3",
    "kernel_size=1",
    "stride=2",
  },
  "3:2:1:5", "conv1d_mb_1x1_kernel.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "dilation=2",
  },
  "1:3:1:11", "conv1d_sb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_mb_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "dilation=2",
  },
  "3:3:1:11", "conv1d_mb_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_same_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "padding=same",
    "dilation=2",
  },
  "1:3:1:11", "conv1d_sb_same_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_mb_same_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {
    "filters=2",
    "kernel_size=3",
    "padding=same",
    "dilation=2",
  },
  "3:3:1:11", "conv1d_mb_same_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_sb_causal_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=2", "padding=causal", "dilation=2"}, "1:1:1:4",
  "conv1d_sb_causal_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

auto conv1d_mb_causal_dilation = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::Conv1DLayer>,
  {"filters=3", "kernel_size=2", "padding=causal", "dilation=2"}, "3:1:1:4",
  "conv1d_mb_causal_dilation.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32");

GTEST_PARAMETER_TEST(
  Convolution1D, LayerGoldenTest,
  ::testing::Values(conv1d_sb_minimum, conv1d_mb_minimum, conv1d_sb_same_remain,
                    conv1d_mb_same_remain, conv1d_sb_same_uneven_remain_1,
                    conv1d_sb_same_uneven_remain_2,
                    conv1d_mb_same_uneven_remain_1,
                    conv1d_mb_same_uneven_remain_2, conv1d_sb_valid_drop_last,
                    conv1d_mb_valid_drop_last, conv1d_sb_no_overlap,
                    conv1d_mb_no_overlap, conv1d_sb_causal, conv1d_mb_causal,
                    conv1d_sb_1x1_kernel, conv1d_mb_1x1_kernel,
                    conv1d_sb_dilation, conv1d_mb_dilation,
                    conv1d_sb_same_dilation, conv1d_mb_same_dilation,
                    conv1d_sb_causal_dilation, conv1d_mb_causal_dilation));
