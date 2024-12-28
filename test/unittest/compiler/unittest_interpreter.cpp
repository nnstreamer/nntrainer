// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_interpreter.cpp
 * @date 02 April 2021
 * @brief interpreter test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <functional>
#include <gtest/gtest.h>
#include <memory>

#include <app_context.h>
#include <ini_interpreter.h>
#include <interpreter.h>
#include <layer.h>
#include <layer_node.h>
#include <network_graph.h>
#include <nntrainer_error.h>

#include <app_context.h>
#include <compiler_test_util.h>
#include <nntrainer_test_util.h>

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;

auto ini_interpreter = std::make_shared<nntrainer::IniGraphInterpreter>(
  nntrainer::Engine::Global(), compilerPathResolver);

/**
 * @brief nntrainer Interpreter Test setup
 *
 * @note Proposing an evolutional path of current test
 * 1. A reference graph vs given parameter
 * 2. A reference graph vs list of models
 * 3. A reference graph vs (pick two models) a -> b -> a graph, b -> a -> b
 * graph
 */
class nntrainerInterpreterTest
  : public ::testing::TestWithParam<
      std::tuple<nntrainer::GraphRepresentation, const char *,
                 std::shared_ptr<nntrainer::GraphInterpreter>>> {

protected:
  virtual void SetUp() {
    auto params = GetParam();

    reference = std::get<0>(params);
    file_path = compilerPathResolver(std::get<1>(params));
    interpreter = std::move(std::get<2>(params));
  }

  nntrainer::GraphRepresentation reference;
  std::shared_ptr<nntrainer::GraphInterpreter> interpreter;
  std::string file_path;
};

/**
 * @brief Check two compiled graph is equal
 * @note later this will be more complicated (getting N graph and compare each
 * other)
 *
 */
TEST_P(nntrainerInterpreterTest, graphEqual) {
  // int status = reference->compile("");
  // EXPECT_EQ(status, ML_ERROR_NONE);
  auto g = interpreter->deserialize(file_path);

  /// @todo: change this to something like graph::finalize
  // status = g->compile("");
  // EXPECT_EQ(status, ML_ERROR_NONE);

  /// @todo: make a proper graph equal
  /// 1. having same number of nodes
  /// 2. layer name is identical (this is too strict though)
  /// 3. attributes of layer is identical
  EXPECT_NO_THROW(graphEqual(g, reference));
}

/**
 * @brief graph serialize after deserialize, compare if they are the same
 *
 */
TEST_P(nntrainerInterpreterTest, graphSerializeAfterDeserialize) {
  auto g = interpreter->deserialize(file_path);

  auto out_file_path = file_path + ".out";

  /// @todo: change this to something like graph::finalize
  // int status = g->compile("");
  // EXPECT_EQ(status, ML_ERROR_NONE);
  interpreter->serialize(g, out_file_path);
  auto new_g = interpreter->deserialize(out_file_path);

  graphEqual(g, new_g);

  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  EXPECT_EQ(remove(out_file_path.c_str()), 0)
    << SAFE_STRERROR(errno, error_buf, error_buflen);
}

TEST_P(nntrainerInterpreterTest, deserialize_01_n) {
  EXPECT_THROW(interpreter->deserialize(""), std::invalid_argument);
}

TEST_P(nntrainerInterpreterTest, deserialize_02_n) {
  EXPECT_THROW(interpreter->deserialize("not_existing_file"),
               std::invalid_argument);
}

auto fc0 = LayerRepresentation("fully_connected",
                               {"name=fc0", "unit=2", "input_shape=1:1:100"});
auto fc1 = LayerRepresentation("fully_connected", {"name=fc1", "unit=2"});

auto flatten = LayerRepresentation("flatten", {"name=flat"});

/**
 * @brief make ini test case from given parameter
 */
static std::tuple<nntrainer::GraphRepresentation, const char *,
                  std::shared_ptr<nntrainer::GraphInterpreter>>
mkTc(nntrainer::GraphRepresentation graph, const char *file,
     std::shared_ptr<nntrainer::GraphInterpreter> interpreter) {
  return std::make_tuple(graph, file, interpreter);
}

// clang-format off
GTEST_PARAMETER_TEST(nntrainerAutoInterpreterTest, nntrainerInterpreterTest,
                        ::testing::Values(
  mkTc(makeGraph({fc0, flatten}), "simple_fc.ini", ini_interpreter),
  mkTc(makeGraph({fc0, flatten}), "simple_fc_backbone.ini", ini_interpreter)
));
// clang-format on
