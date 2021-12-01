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

static std::unique_ptr<NeuralNetwork> makeMolAttention() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=in3", "input_shape=1:1:5"}},
    {"input", {"name=in2", "input_shape=1:4:6"}},
    {"input", {"name=in1", "input_shape=1:1:6"}},
    {"mol_attention",
     {"name=mol", "input_layers=in1,in2,in3", "unit=8", "mol_k=5"}},
    {"constant_derivative", {"name=loss", "input_layers=mol"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

INSTANTIATE_TEST_CASE_P(
  model, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelIniTc(reduce_mean_last, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeMolAttention, "mol_attention",
                 ModelTestOption::COMPARE_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info) {
    return std::get<1>(info.param);
  });
