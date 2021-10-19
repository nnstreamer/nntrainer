// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_models_recurrent.cpp
 * @date 05 Oct 2021
 * @brief unittest models for recurrent ones
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <databuffer.h>
#include <dataset.h>
#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

static inline constexpr const int NOT_USED_ = 1;

static IniSection nn_base("model", "type = NeuralNetwork");
static std::string fc_base = "type = Fully_connected";
static IniSection sgd_base("optimizer", "Type = sgd");
static IniSection constant_loss("loss", "type = constant_derivative");

IniWrapper fc_unroll_single(
  "fc_unroll_single",
  {
    nn_base,
    sgd_base + "learning_rate=0.1",
    IniSection("fc_1") + fc_base + "unit=1 | input_shape=1:1:1",
    IniSection("fc_2") + fc_base + "unit=1 | shared_from = fc_1",
    IniSection("fc_3") + fc_base + "unit=1 | shared_from = fc_1",
    IniSection("fc_4") + fc_base + "unit=1 | shared_from = fc_1",
    IniSection("fc_5") + fc_base + "unit=1 | shared_from = fc_1",
    constant_loss,
  });

std::unique_ptr<NeuralNetwork> makeFC() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=1"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:1"}},
    /// here lstm_cells is being inserted
    {"constant_derivative", {"name=loss", "input_layers=recurrent/a2"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto fcfc = makeGraph({
    {"Fully_connected", {"name=a1", "unit=1"}},
    {"Fully_connected", {"name=a2", "unit=1", "input_layers=a1"}},
  });

  nn->addWithReferenceLayers(fcfc, "recurrent", {"input"}, {"a1"}, {"a2"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "return_sequences=false",
                               "recurrent_input=a1",
                               "recurrent_output=a2",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSingleLSTM() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here lstm_cells is being inserted
    {"mse", {"name=loss", "input_layers=lstm_scope/a1"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto lstm = makeGraph({
    {"lstm", {"name=a1", "unit=2"}},
  });

  nn->addWithReferenceLayers(lstm, "lstm_scope", {"input"}, {"a1"}, {"a1"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "return_sequences=true",
                               "recurrent_input=a1",
                               "recurrent_output=a1",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeStackedLSTM() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here lstm_cells is being inserted
    {"mse", {"name=loss", "input_layers=lstm_scope/a2"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto lstm = makeGraph({
    {"lstm", {"name=a1", "unit=2"}},
    {"lstm", {"name=a2", "unit=2", "input_layers=a1"}},
  });

  nn->addWithReferenceLayers(lstm, "lstm_scope", {"input"}, {"a1"}, {"a2"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "return_sequences=true",
                               "recurrent_input=a1",
                               "recurrent_output=a2",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

INSTANTIATE_TEST_CASE_P(
  recurrentModels, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelIniTc(fc_unroll_single, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeFC, "fc_unroll_stacked", ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleLSTM, "lstm_single", ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedLSTM, "lstm_stacked", ModelTestOption::COMPARE_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info) {
    return std::get<1>(info.param);
  });
