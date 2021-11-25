// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_models_v2.cpp
 * @date 25 Nov 2021
 * @brief unittest models for v2 version
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

static inline constexpr const int NOT_USED_ = 1;

static IniSection nn_base("model", "type = NeuralNetwork");
static std::string fc_base = "type = Fully_connected";
static std::string red_mean_base = "type = reduce_mean";
static IniSection sgd_base("optimizer", "Type = sgd");
static IniSection constant_loss("loss", "type = constant_derivative");

IniWrapper reduce_mean_last("reduce_mean_last",
                            {
                              nn_base + "batch_size=3",
                              sgd_base + "learning_rate=0.1",
                              IniSection("fc_1") + fc_base +
                                "unit=7 | input_shape=1:1:2",
                              IniSection("red_mean") + red_mean_base + "axis=3",
                              constant_loss,
                            });

INSTANTIATE_TEST_CASE_P(
  model, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelIniTc(reduce_mean_last, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info) {
    return std::get<1>(info.param);
  });
