// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file unittest_layers_fully_connected_cl.cpp
 * @date 7 June 2024
 * @brief Fully Connected Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <fused_fc_norm_cl.h>
#include <layers_common_tests.h>
#include <nntrainer_log.h>

auto semantic_fused_gpu = LayerSemanticsParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedRMSNormLayerCl>,
  nntrainer::FullyConnectedRMSNormLayerCl::type, {"unit=1"},
  LayerCreateSetPropertyOptions::AVAILABLE_FROM_APP_CONTEXT, false, 1);

GTEST_PARAMETER_TEST(FusedLayerGPU, LayerSemanticsGpu,
                     ::testing::Values(semantic_fused_gpu));

auto fused_gpu_plain = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::FullyConnectedRMSNormLayerCl>, {"unit=3"},
  "1:1:2:3", "fused_plain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT,
  "nchw", "fp32", "fp32");

// auto fused_gpu_plain_Ind = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedRMSNormLayerCl>, {"unit=1",
//   "epsilon=0.001"}, "1:1:2:3", "fused_plain_Ind.nnlayergolden",
//   LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

// auto fused_gpu_single_batch = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedRMSNormLayerCl>, {"unit=2",
//   "epsilon=0.001"}, "1:1:1:3", "fused_single_batch1.nnlayergolden",
//   LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp32", "fp32");

// auto fc_gpu_no_decay = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>,
//   {"unit=5", "weight_decay=0.0", "bias_decay=0.0"}, "3:1:1:10",
//   "fc_plain.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT, "nchw",
//   "fp32", "fp32");

// auto fc_gpu_plain_nhwc = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>, {"unit=5"},
//   "3:10:1:1", "fc_plain.nnlayergolden",
//   LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//     LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
//     LayerGoldenTestParamOptions::USE_INC_FORWARD,
//   "nhwc", "fp32", "fp32");

// auto fc_gpu_single_batch_nhwc = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>, {"unit=4"},
//   "1:10:1:1", "fc_single_batch.nnlayergolden",
//   LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//     LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
//   "nhwc", "fp32", "fp32");

// auto fc_gpu_no_decay_nhwc = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>,
//   {"unit=5", "weight_decay=0.0", "bias_decay=0.0"}, "3:10:1:1",
//   "fc_plain.nnlayergolden",
//   LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//     LayerGoldenTestParamOptions::SKIP_CALC_GRAD,
//   "nhwc", "fp32", "fp32");

GTEST_PARAMETER_TEST(FusedLayerGPU, LayerGoldenTest,
                     ::testing::Values(fused_gpu_plain));

// #ifdef ENABLE_FP16
// auto fc_gpu_basic_plain_w16a16 = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>, {"unit=5"},
//   "3:1:1:10", "fc_plain_w16a16.nnlayergolden",
//   LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

// auto fc_gpu_basic_single_batch_w16a16 = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>, {"unit=4"},
//   "1:1:1:10", "fc_single_batch_w16a16.nnlayergolden",
//   LayerGoldenTestParamOptions::DEFAULT, "nchw", "fp16", "fp16");

// auto fc_gpu_basic_no_decay_w16a16 = LayerGoldenTestParamType(
//   nntrainer::createLayer<nntrainer::FullyConnectedLayerCl>,
//   {"unit=5", "weight_decay=0.0", "bias_decay=0.0"}, "3:1:1:10",
//   "fc_plain_w16a16.nnlayergolden",
//   LayerGoldenTestParamOptions::SKIP_CALC_DERIV |
//     LayerGoldenTestParamOptions::SKIP_CALC_GRAD |
//     LayerGoldenTestParamOptions::USE_INC_FORWARD,
//   "nchw", "fp16", "fp16");

// GTEST_PARAMETER_TEST(FullyConnectedGPU16, LayerGoldenTest,
//                      ::testing::Values(fc_gpu_basic_plain_w16a16,
//                                        fc_gpu_basic_single_batch_w16a16,
//                                        fc_gpu_basic_no_decay_w16a16));
// #endif
