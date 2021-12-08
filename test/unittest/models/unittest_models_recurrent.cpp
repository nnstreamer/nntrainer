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

IniWrapper fc_unroll_single__1(
  "fc_unroll_single__1",
  {
    nn_base,
    sgd_base + "learning_rate=0.1",
    IniSection("fc_1") + fc_base +
      "unit=1 | shared_from = fc_2 |input_shape=1:1:1",
    IniSection("fc_2") + fc_base + "unit=1 | shared_from = fc_2",
    IniSection("fc_3") + fc_base + "unit=1 | shared_from = fc_2",
    IniSection("fc_4") + fc_base + "unit=1 | shared_from = fc_2",
    IniSection("fc_5") + fc_base + "unit=1 | shared_from = fc_2",
    constant_loss,
  });

IniWrapper fc_unroll_single__2(
  "fc_unroll_single__2",
  {
    nn_base,
    sgd_base + "learning_rate=0.1",
    IniSection("fc_1") + fc_base +
      "unit=1 | input_shape=1:1:1 | clip_grad_by_norm = 10000.0",
    IniSection("fc_2") + fc_base +
      "unit=1 | shared_from = fc_1 | clip_grad_by_norm = 10000.0",
    IniSection("fc_3") + fc_base +
      "unit=1 | shared_from = fc_1 | clip_grad_by_norm = 10000.0",
    IniSection("fc_4") + fc_base +
      "unit=1 | shared_from = fc_1 | clip_grad_by_norm = 10000.0",
    IniSection("fc_5") + fc_base +
      "unit=1 | shared_from = fc_1 | clip_grad_by_norm = 10000.0",
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
                               "recurrent_input=a1",
                               "recurrent_output=a2",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

std::unique_ptr<NeuralNetwork> makeFCClipped() {
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
    {"Fully_connected", {"name=a1", "unit=1", "clip_grad_by_norm=0.0001"}},
    {"Fully_connected",
     {"name=a2", "unit=1", "input_layers=a1", "clip_grad_by_norm=0.0001"}},
  });

  nn->addWithReferenceLayers(fcfc, "recurrent", {"input"}, {"a1"}, {"a2"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
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
    /// here lstm is being inserted
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
                               "as_sequence=a1",
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
    /// here lstm is being inserted
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
                               "as_sequence=a2",
                               "recurrent_input=a1",
                               "recurrent_output=a2",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSingleLSTMCell() {
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
    {"lstmcell", {"name=a1", "unit=2"}},
  });

  nn->addWithReferenceLayers(lstm, "lstm_scope", {"input"}, {"a1"}, {"a1"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a1",
                               "recurrent_input=a1",
                               "recurrent_output=a1",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeStackedLSTMCell() {
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
    {"lstmcell", {"name=a1", "unit=2"}},
    {"lstmcell", {"name=a2", "unit=2", "input_layers=a1"}},
  });

  nn->addWithReferenceLayers(lstm, "lstm_scope", {"input"}, {"a1"}, {"a2"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a2",
                               "recurrent_input=a1",
                               "recurrent_output=a2",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSingleZoneoutLSTMCell() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=1"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here zoneout_lstm_cell is being inserted
    {"mse", {"name=loss", "input_layers=zoneout_lstm_scope/a1"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto zoneout_lstm = makeGraph({
    {"zoneout_lstmcell",
     {"name=a1", "unit=2", "hidden_state_zoneout_rate=1.0",
      "cell_state_zoneout_rate=1.0", "test=true"}},
  });

  nn->addWithReferenceLayers(zoneout_lstm, "zoneout_lstm_scope", {"input"},
                             {"a1"}, {"a1"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a1",
                               "recurrent_input=a1",
                               "recurrent_output=a1",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeStackedZoneoutLSTMCell() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=1"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here zoneout_lstm_cell is being inserted
    {"mse", {"name=loss", "input_layers=zoneout_lstm_scope/a2"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto zoneout_lstm = makeGraph({
    {"zoneout_lstmcell",
     {"name=a1", "unit=2", "hidden_state_zoneout_rate=1.0",
      "cell_state_zoneout_rate=1.0", "test=true"}},
    {"zoneout_lstmcell",
     {"name=a2", "unit=2", "hidden_state_zoneout_rate=1.0",
      "cell_state_zoneout_rate=1.0", "test=true", "input_layers=a1"}},
  });

  nn->addWithReferenceLayers(zoneout_lstm, "zoneout_lstm_scope", {"input"},
                             {"a1"}, {"a2"},
                             ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a2",
                               "recurrent_input=a1",
                               "recurrent_output=a2",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSingleRNNCell() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here rnncell is being inserted
    {"mse", {"name=loss", "input_layers=rnncell_scope/a1"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto rnncell = makeGraph({
    {"rnncell", {"name=a1", "unit=2", "integrate_bias=false"}},
  });

  nn->addWithReferenceLayers(rnncell, "rnncell_scope", {"input"}, {"a1"},
                             {"a1"}, ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a1",
                               "recurrent_input=a1",
                               "recurrent_output=a1",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeStackedRNNCell() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here rnncells are being inserted
    {"mse", {"name=loss", "input_layers=rnncell_scope/a2"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto rnncell = makeGraph({
    {"rnncell", {"name=a1", "unit=2", "integrate_bias=false"}},
    {"rnncell",
     {"name=a2", "unit=2", "integrate_bias=false", "input_layers=a1"}},
  });

  nn->addWithReferenceLayers(rnncell, "rnncell_scope", {"input"}, {"a1"},
                             {"a2"}, ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a2",
                               "recurrent_input=a1",
                               "recurrent_output=a2",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeSingleGRUCell() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here grucell is being inserted
    {"mse", {"name=loss", "input_layers=grucell_scope/a1"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto grucell = makeGraph({
    {"grucell", {"name=a1", "unit=2"}},
  });

  nn->addWithReferenceLayers(grucell, "grucell_scope", {"input"}, {"a1"},
                             {"a1"}, ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a1",
                               "recurrent_input=a1",
                               "recurrent_output=a1",
                             });

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeStackedGRUCell() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=1:1:2"}},
    /// here grucells are being inserted
    {"mse", {"name=loss", "input_layers=grucell_scope/a2"}},
  });
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  auto grucell = makeGraph({
    {"grucell", {"name=a1", "unit=2"}},
    {"grucell", {"name=a2", "unit=2", "input_layers=a1"}},
  });

  nn->addWithReferenceLayers(grucell, "grucell_scope", {"input"}, {"a1"},
                             {"a2"}, ml::train::ReferenceLayersType::RECURRENT,
                             {
                               "unroll_for=2",
                               "as_sequence=a2",
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
    mkModelIniTc(fc_unroll_single__1, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelIniTc(fc_unroll_single__2, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeFC, "fc_unroll_stacked", ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeFCClipped, "fc_unroll_stacked_clipped",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleLSTM, "lstm_single", ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleLSTMCell, "lstm_single__1",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedLSTM, "lstm_stacked", ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedLSTMCell, "lstm_stacked__1",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_000_000",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_000_000",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_050_000",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_050_000",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_100_000",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_100_000",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_000_050",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_000_050",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_050_050",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_050_050",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_100_050",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_100_050",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_000_100",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_000_100",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_050_100",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_050_100",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleZoneoutLSTMCell, "zoneout_lstm_single_100_100",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedZoneoutLSTMCell, "zoneout_lstm_stacked_100_100",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleRNNCell, "rnncell_single__1",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedRNNCell, "rnncell_stacked__1",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeSingleGRUCell, "grucell_single__1",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeStackedGRUCell, "grucell_stacked__1",
                 ModelTestOption::COMPARE_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info) {
    return std::get<1>(info.param);
  });
