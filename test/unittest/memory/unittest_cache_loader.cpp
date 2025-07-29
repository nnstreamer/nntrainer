// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file unittest_cache_loader.cpp
 * @date 14 Nov 2022
 * @brief Cache Loader Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include "task_executor.h"
#include <cstring>

#include <future>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cache_loader.h>
#include <cache_pool.h>
#include <nntrainer_test_util.h>
#include <optimized_v1_planner.h>
#include <optimized_v2_planner.h>
#include <optimized_v3_planner.h>

/**
 * @brief Cache loader test class
 */
class CacheLoaderTest
  : public ::testing::TestWithParam<std::shared_ptr<nntrainer::MemoryPlanner>> {
public:
  void SetUp(void) {
    pool = std::make_shared<nntrainer::CachePool>("tmp pool");
    loader = new nntrainer::CacheLoader(pool);
    loader->init();
    planner = GetParam();
  }

  void TearDown(void) {
    loader->finish();
    delete loader;
  }

  std::shared_ptr<nntrainer::CachePool> pool;
  nntrainer::CacheLoader *loader;
  std::shared_ptr<nntrainer::MemoryPlanner> planner;
};

/**
 * @brief load synchronously
 */
TEST_P(CacheLoaderTest, load_01_p) {
  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  EXPECT_NO_THROW(pool->planLayout(*planner.get()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 4);
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);
  EXPECT_EQ(mem->getAddr(), nullptr);

  // Check memory loading for each execution order
  loader->load(1);
  EXPECT_NE(mem->getAddr(), nullptr);
  pool->flush();

  loader->load(2);
  EXPECT_NE(mem->getAddr(), nullptr);
  pool->flush();

  loader->load(3);
  EXPECT_NE(mem->getAddr(), nullptr);
  pool->flush();

  loader->load(4);
  EXPECT_NE(mem->getAddr(), nullptr);
  pool->flush();

  loader->load(5);
  EXPECT_NE(mem->getAddr(), nullptr);
  pool->flush();

  loader->load(6);
  EXPECT_EQ(mem->getAddr(), nullptr);
  pool->flush();
}

/**
 * @brief load synchronously multiple
 */
TEST_P(CacheLoaderTest, load_02_p) {
  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 5, 6, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});
  EXPECT_NO_THROW(pool->planLayout(*planner.get()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  // Check memory loading for each execution order
  loader->load(1);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(2);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(3);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(4);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(5);
  EXPECT_NE(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(6);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(7);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(8);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  pool->flush();

  loader->load(9);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);
  pool->flush();
}

/**
 * @brief load asynchronously
 */
TEST_P(CacheLoaderTest, load_async_01_p) {
  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  EXPECT_NO_THROW(pool->planLayout(*planner.get()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 4);
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);
  EXPECT_EQ(mem->getAddr(), nullptr);

  std::promise<void> p;
  auto future = p.get_future();
  auto complete = [&p, &mem](int id,
                             nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem->getAddr(), nullptr);
    p.set_value();
  };

  // Check memory loading for each execution order
  int id = loader->loadAsync(1, complete);

  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem->getAddr(), nullptr);
}

/**
 * @brief load asynchronously
 */
TEST_P(CacheLoaderTest, load_async_02_p) {
  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 5, 6, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});
  EXPECT_NO_THROW(pool->planLayout(*planner.get()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  // Check memory loading for execution order(1)
  auto p = new std::promise<void>;
  auto future = p->get_future();
  auto complete1 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  int id = loader->loadAsync(1, complete1);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(2)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete2 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_NE(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(2, complete2);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(3)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete3 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_NE(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(3, complete3);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(4)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete4 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_NE(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(4, complete4);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(5)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete5 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(5, complete5);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(6)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete6 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(6, complete6);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(7)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete7 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(7, complete7);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(8)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete8 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(8, complete8);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(9)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete9 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(9, complete9);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;
}

/**
 * @brief load asynchronously (discontinous order)
 */
TEST_P(CacheLoaderTest, load_async_03_p) {
  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});
  EXPECT_NO_THROW(pool->planLayout(*planner.get()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  // Check memory loading for execution order(1)
  auto p = new std::promise<void>;
  auto future = p->get_future();
  auto complete1 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  int id = loader->loadAsync(1, complete1);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(2)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete2 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_NE(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(2, complete2);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(3)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete3 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_NE(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(3, complete3);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(4)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete4 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_NE(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(4, complete4);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(5)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete5 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_NE(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(5, complete5);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(6)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete6 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(6, complete6);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(7)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete7 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(7, complete7);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(8)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete8 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_NE(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(8, complete8);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;

  // Check memory loading for execution order(9)
  p = new std::promise<void>;
  future = p->get_future();

  auto complete9 = [&](int id, nntrainer::TaskExecutor::CompleteStatus status) {
    EXPECT_EQ(status, nntrainer::TaskExecutor::SUCCESS);
    EXPECT_EQ(mem1->getAddr(), nullptr);
    EXPECT_EQ(mem2->getAddr(), nullptr);
    EXPECT_EQ(mem3->getAddr(), nullptr);
    p->set_value();
  };

  id = loader->loadAsync(9, complete9);
  future.wait();
  pool->flush();

  EXPECT_NE(id, 0);
  EXPECT_EQ(mem1->getAddr(), nullptr);
  EXPECT_EQ(mem2->getAddr(), nullptr);
  EXPECT_EQ(mem3->getAddr(), nullptr);

  delete p;
}

GTEST_PARAMETER_TEST(
  CacheLoader, CacheLoaderTest,
  ::testing::Values(std::make_shared<nntrainer::BasicPlanner>(),
                    std::make_shared<nntrainer::OptimizedV1Planner>(),
                    std::make_shared<nntrainer::OptimizedV2Planner>(),
                    std::make_shared<nntrainer::OptimizedV3Planner>()));

/**
 * TODO: cancel and timeout logic test
 */
