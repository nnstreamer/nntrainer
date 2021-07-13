// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_batch_queue.cpp
 * @date 12 July 2021
 * @brief Batch Queue Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <batch_queue.h>
#include <tensor.h>

#include <future>
#include <thread>
#include <tuple>
#include <vector>

nntrainer::DataProducer::Iteration data(size_t key) {
  return {true, std::vector<nntrainer::Tensor>(key), {}};
};

void test_data(const nntrainer::DataProducer::Iteration &dat,
               size_t expected_key) {
  EXPECT_EQ(std::get<1>(dat).size(), expected_key);
}

TEST(BatchQueue, pushPop_p) {
  nntrainer::BatchQueue bq(1);

  EXPECT_NO_THROW(bq.wait_and_push(data(1)));
  auto result = bq.wait_and_pop();
  test_data(*result, 1);
}

TEST(BatchQueue, threadedPushPops_p) {
  /** preparing primitives */
  using namespace std::chrono_literals;
  auto push_after = [](nntrainer::BatchQueue &bq, const auto &duration,
                       size_t key) {
    std::this_thread::sleep_for(duration);
    EXPECT_NO_THROW(bq.wait_and_push(data(key)));
  };
  auto pop_after = [](nntrainer::BatchQueue &bq, const auto &duration,
                      size_t key) {
    std::this_thread::sleep_for(duration);
    auto result = bq.wait_and_pop();
    test_data(*result, key);
  };

  std::vector<std::future<void>> futures;
  {
    futures.clear();
    /// 0     -> push(1)
    /// 250ms -> pop(1)
    nntrainer::BatchQueue bq(1);
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 0ms, 1));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 250ms, 1));
    for (auto &future : futures) {
      future.get();
    }
  }

  {
    futures.clear();
    /// 0     -> pop(1)
    /// 250ms -> push(1)
    nntrainer::BatchQueue bq(1);
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 0ms, 1));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 250ms, 1));
    for (auto &future : futures) {
      future.get();
    }
  }

  {
    futures.clear();
    /// 0     -> push(1)
    /// 300ms -> push(2)
    /// 300ms -> pop(1)
    /// 500ms -> push(3)
    /// 600ms -> push(4) (waits)
    /// 750ms -> pop(2)
    /// 1000ms-> pop(3)
    nntrainer::BatchQueue bq(2);
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 0ms, 1));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 300ms, 2));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 300ms, 1));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 500ms, 3));
    futures.push_back(
      std::async(std::launch::async, push_after, std::ref(bq), 600ms, 4));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 750ms, 2));
    futures.push_back(
      std::async(std::launch::async, pop_after, std::ref(bq), 1000ms, 3));
    for (auto &future : futures) {
      future.get();
    }
  }
}
