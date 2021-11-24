// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_models_multiout.cpp
 * @date 22 November 2021
 * @brief unittest models to cover multiinput, multioutput scenario
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

/// A has two output tensor a1, a2 and B, C takes it
///     A
/// (a0, a1)
///  |    |
///  v    v
/// (a0) (a1)
///  B    C
///   \  /
///    v
///    D
static std::unique_ptr<NeuralNetwork> split_and_join() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=5"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:3", "unit=2"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"fully_connected", {"name=c", "input_layers=a(1)", "unit=3"}},
    {"fully_connected", {"name=b", "input_layers=a(0)", "unit=3"}},
    {"addition", {"name=d", "input_layers=b,c"}},
    {"mse", {"name=loss", "input_layers=d"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

/// A has two output tensor a1, a2 and B takes it
///     A
/// (a0, a1)
///  |    |
///  v    v
/// (a0, a1)
///    B
static std::unique_ptr<NeuralNetwork> one_to_one() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=5"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:3", "unit=2"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"addition", {"name=b", "input_layers=a(0),a(1)"}},
    {"mse", {"name=loss", "input_layers=b"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

/// A has two output tensor a0, aa and B takes it but in reversed order
///     A
/// (a0, a1)
///     x
///  v    v
/// (a0, a1)
///    B
static std::unique_ptr<NeuralNetwork> one_to_one_reversed() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=5"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:3", "unit=2"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"addition", {"name=b", "input_layers=a(1),a(0)"}},
    {"mse", {"name=loss", "input_layers=b"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

/// A has two output tensor a1, a2 and B and C takes it
///     A
/// (a0, a1, a2)---------------->(a2)
///  | \  |-------                E
///  |  - + - \   |              /
///  v    v   v   v             /
/// (a0, a1) (a0, a1)          /
///    B         C            /
///   (b0)      (c0)         /
///     \        /          /
///      \      /----------
///         v
///        (d0)
///          D
static std::unique_ptr<NeuralNetwork> one_to_many() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=5"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:2", "unit=3"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"addition", {"name=b", "input_layers=a(0),a(1)"}},
    {"addition", {"name=c", "input_layers=a(0),a(1)"}},
    {"addition", {"name=d", "input_layers=b,c,a(2)"}},
    {"mse", {"name=loss", "input_layers=d"}},
  });

  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

INSTANTIATE_TEST_CASE_P(
  multiInoutModels, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelTc_V2(split_and_join, "split_and_join", ModelTestOption::ALL_V2),
    mkModelTc_V2(one_to_one, "one_to_one", ModelTestOption::ALL_V2),
    mkModelTc_V2(one_to_one_reversed, "one_to_one__reversed",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(one_to_many, "one_to_many", ModelTestOption::ALL_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info) {
    return std::get<1>(info.param);
  });
