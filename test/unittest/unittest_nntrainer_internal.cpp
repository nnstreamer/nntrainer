/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        unittest_nntrainer_internal.cpp
 * @date        10 April 2020
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <thread>

#include <bs_thread_pool_manager.hpp>
#include <engine.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <optimizer.h>
#include <util_func.h>

#include <nntrainer_test_util.h>

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_01_p) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_NO_THROW(op = ac->createOptimizerObject("adam", {}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_02_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("adam", {"unknown"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_03_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("adam", {"lr=0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_04_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op =
                     ac->createOptimizerObject("adam", {"learning_rate:0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_05_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_NO_THROW(op = ac->createOptimizerObject("sgd", {}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_06_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("sgd", {"lr=0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_07_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  // auto &ac = nntrainer::AppContext::Global();
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op =
                     ac->createOptimizerObject("sgd", {"learning_rate:0.1"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_08_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("sgd", {"unknown"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_09_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("non-existing type", {}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_adamw_01_p) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_NO_THROW(op =
                    ac->createOptimizerObject("adamw", {"weight_decay=0.01"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_adamw_02_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op = ac->createOptimizerObject("adamw", {"unknown"}));
}

/**
 * @brief Optimizer create
 */
TEST(nntrainer_Optimizer, create_adamw_03_n) {
  std::unique_ptr<nntrainer::Optimizer> op;
  auto &eg = nntrainer::Engine::Global();
  auto ac = eg.getRegisteredContext("cpu");
  EXPECT_ANY_THROW(op =
                     ac->createOptimizerObject("adamw", {"learning_rate:0.1"}));
}

TEST(nntrainer_throw_if, throw_invalid_arg_p) {
  try {
    NNTR_THROW_IF(1 == 1, std::invalid_argument) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
  }

  try {
    NNTR_THROW_IF(true, std::invalid_argument) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
  }

  bool hit = false;
  auto cleanup = [&hit] { hit = true; };
  try {
    NNTR_THROW_IF_CLEANUP(true, std::invalid_argument, cleanup) << "error msg";
  } catch (std::invalid_argument &e) {
    EXPECT_STREQ("error msg", e.what());
    EXPECT_TRUE(hit);
  }
}

TEST(nntrainer_engine, getFullPath_handles_edge_cases) {
  EXPECT_EQ(".", nntrainer::getFullPath("", ""));

  const auto base = std::filesystem::current_path().string();
  EXPECT_EQ(base, nntrainer::getFullPath("", base));
}

TEST(nntrainer_engine, getFullPath_resolves_relative_and_absolute) {
  const auto absolute = std::filesystem::current_path().string();
  EXPECT_EQ(absolute, nntrainer::getFullPath(absolute, "/invalid/base"));

  const std::string relative_name = "dummy_plugin.so";
  const auto expected =
    (std::filesystem::current_path() / std::filesystem::path(relative_name))
      .string();
  EXPECT_EQ(expected, nntrainer::getFullPath(relative_name, absolute));
}

TEST(nntrainer_engine, getFullPath_empty_path_relative_base_n) {
  const std::string relative_base = "relative/plugins";
  EXPECT_EQ(relative_base, nntrainer::getFullPath("", relative_base));
}

TEST(nntrainer_engine, getFullPath_empty_path_current_dir_base_n) {
  const std::string base = ".";
  EXPECT_EQ(base, nntrainer::getFullPath("", base));
}

TEST(nntrainer_engine, getFullPath_empty_path_trailing_slash_base_n) {
  const std::string base = "/tmp/nntrainer/";
  EXPECT_EQ(base, nntrainer::getFullPath("", base));
}

TEST(nntrainer_thread_pool_manager, select_single_thread_for_small_workload) {
  auto &manager = nntrainer::ThreadPoolManager::Global();
  EXPECT_EQ(1u, manager.select_k_quant_thread_count(1, 1, 1));
}

TEST(nntrainer_thread_pool_manager, respect_medium_workload_threshold) {
  auto &manager = nntrainer::ThreadPoolManager::Global();
  const std::size_t max_threads =
    std::max<std::size_t>(1U, std::thread::hardware_concurrency());
  const std::size_t expected = std::min<std::size_t>(2U, max_threads);
  EXPECT_EQ(expected, manager.select_k_quant_thread_count(1536, 1536, 1));
}

TEST(nntrainer_thread_pool_manager, zero_dimension_clamps_to_single_thread_n) {
  auto &manager = nntrainer::ThreadPoolManager::Global();
  EXPECT_EQ(1u, manager.select_k_quant_thread_count(0, 2048, 4096));
}

TEST(nntrainer_thread_pool_manager,
     near_first_threshold_stays_single_thread_n) {
  auto &manager = nntrainer::ThreadPoolManager::Global();
  EXPECT_EQ(1u, manager.select_k_quant_thread_count(1200, 1500, 1));
}

TEST(nntrainer_thread_pool_manager,
     second_band_does_not_jump_to_four_threads_n) {
  auto &manager = nntrainer::ThreadPoolManager::Global();
  const auto hw_threads =
    std::max<std::size_t>(1U, std::thread::hardware_concurrency());
  if (hw_threads < 4U) {
    GTEST_SKIP() << "Requires >=4 hardware threads to validate regression";
  }

  EXPECT_EQ(2u, manager.select_k_quant_thread_count(1536, 1800, 1));
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
