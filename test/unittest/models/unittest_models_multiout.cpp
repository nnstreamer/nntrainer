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
  nn->setProperty({"batch_size=1"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:2", "unit=2"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"fully_connected", {"name=b", "input_layers=a(0)", "unit=2"}},
    {"fully_connected", {"name=c", "input_layers=a(1)", "unit=2"}},
    {"addition", {"name=d", "input_layers=b,c"}},
    {"constant_derivative", {"name=loss", "input_layers=d"}},
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
[[maybe_unused]] static std::unique_ptr<NeuralNetwork> one_to_one() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=1"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:2", "unit=2"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"addition", {"name=b", "input_layers=a(0),a(1)"}},
    {"constant_derivative", {"name=loss", "input_layers=b"}},
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
[[maybe_unused]] static std::unique_ptr<NeuralNetwork> one_to_one_reversed() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=1"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:2", "unit=2"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"addition", {"name=b", "input_layers=a(1),a(0)"}},
    {"constant_derivative", {"name=loss", "input_layers=b"}},
  });
  for (auto &node : graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

/// A has two output tensor a1, a2 and B and C takes it
///     A
/// (a0, a1)
///  | \  |-------
///  |  - + - \   |
///  v    v   v   v
/// (a0, a1) (a0, a1)
///    B         C
///   (b0)      (c0)
///     \        /
///      \      /
///         v
///        (d0)
///          D
[[maybe_unused]] static std::unique_ptr<NeuralNetwork> one_to_many() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=1"});

  auto graph = makeGraph({
    {"fully_connected", {"name=fc", "input_shape=1:1:2", "unit=2"}},
    {"split", {"name=a", "input_layers=fc", "axis=3"}},
    {"addition", {"name=b", "input_layers=a(0),a(1)"}},
    {"addition", {"name=c", "input_layers=a(0),a(1)"}},
    {"addition", {"name=d", "input_layers=b,c"}},
    {"constant_derivative", {"name=loss", "input_layers=d"}},
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
    mkModelTc_V2(split_and_join, "split_and_join",
                 ModelTestOption::SAVE_AND_LOAD_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info) {
    return std::get<1>(info.param);
  });
