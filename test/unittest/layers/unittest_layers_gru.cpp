// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_layers_gru.cpp
 * @date 11 July 2021
 * @brief GRU Layer Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>

#include <gru.h>
#include <layers_common_tests.h>

auto semantic_gru =
  LayerSemanticsParamType(nntrainer::createLayer<nntrainer::GRULayer>,
                          nntrainer::GRULayer::type, {"unit=1"}, 0, false, 1);

INSTANTIATE_TEST_CASE_P(GRU, LayerSemantics, ::testing::Values(semantic_gru));

auto gru_single_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRULayer>, {"unit=5"}, "3:1:1:7",
  "gru_single_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto gru_multi_step = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRULayer>, {"unit=5"}, "3:1:4:7",
  "gru_multi_step.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto gru_single_step_seq = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRULayer>,
  {"unit=5", "return_sequences=true"}, "3:1:1:7",
  "gru_single_step_seq.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto gru_multi_step_seq = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRULayer>,
  {"unit=5", "return_sequences=true"}, "3:1:4:7",
  "gru_multi_step_seq.nnlayergolden", LayerGoldenTestParamOptions::DEFAULT);

auto gru_multi_step_seq_act_orig = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRULayer>,
  {"unit=5", "return_sequences=true", "hidden_state_activation=tanh",
   "recurrent_activation=sigmoid"},
  "3:1:4:7", "gru_multi_step_seq.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

auto gru_multi_step_seq_act = LayerGoldenTestParamType(
  nntrainer::createLayer<nntrainer::GRULayer>,
  {"unit=5", "return_sequences=true", "hidden_state_activation=sigmoid",
   "recurrent_activation=tanh"},
  "3:1:4:7", "gru_multi_step_seq_act.nnlayergolden",
  LayerGoldenTestParamOptions::DEFAULT);

INSTANTIATE_TEST_CASE_P(GRU, LayerGoldenTest,
                        ::testing::Values(gru_single_step, gru_multi_step,
                                          gru_single_step_seq,
                                          gru_multi_step_seq,
                                          gru_multi_step_seq_act_orig,
                                          gru_multi_step_seq_act));
