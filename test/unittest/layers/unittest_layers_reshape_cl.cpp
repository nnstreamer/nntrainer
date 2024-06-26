// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   unittest_layers_reshape_cl.cpp
 * @date   18 June 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Reshape Layer Test
 */
#include <tuple>

#include <gtest/gtest.h>

#include <layers_common_tests.h>
#include <reshape_cl.h>

auto semantic_reshape_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::ReshapeLayerCl>,
  nntrainer::ReshapeLayerCl::type, {"target_shape=-1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(ReshapeGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_reshape_gpu));

auto reshape_basic_plain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::ReshapeLayerCl>, {"target_shape=-1"},
  "2:3:3:3", "reshape.nnlayergolden",
  LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
    LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
    LayerGoldenTestParamOptions::USE_INC_FORWARD,
  "nchw", "fp32", "fp32");

GTEST_PARAMETER_TEST(ReshapeGPU, LayerGoldenTest,
                     ::testing::Values(reshape_basic_plain));
