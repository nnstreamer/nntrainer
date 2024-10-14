// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file unittest_cache_pool.cpp
 * @date 28 Sep 2022
 * @brief Cache Pool Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include "optimized_v1_planner.h"
#include <cstring>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <basic_planner.h>
#include <cache_pool.h>
#include <nntrainer_test_util.h>

constexpr float TEMP_DATA1 = (12345.12345);
constexpr float TEMP_DATA2 = (67890.67890);
constexpr float TEMP_DATA3 = (54321.54321);

/**
 * @brief Mock class for cache pool
 */
class MockCachePool : public nntrainer::CachePool {
public:
  MockCachePool(const std::string &name) : nntrainer::CachePool(name) {
    ON_CALL(*this, validate).WillByDefault([&](unsigned int id) {
      nntrainer::CachePool::validate(id);
    });
    ON_CALL(*this, invalidate).WillByDefault([&](unsigned int id) {
      nntrainer::CachePool::invalidate(id);
    });
  }

  /**
   * @brief Mock method for validate
   */
  MOCK_METHOD(void, validate, (unsigned int id), (override));

  /**
   * @brief Mock method for invalidate
   */
  MOCK_METHOD(void, invalidate, (unsigned int id), (override));
};

/**
 * @brief Cache pool test class
 */
class CachePoolTest : public ::testing::Test {
public:
  void SetUp(void) { pool = new MockCachePool("tmp pool"); }

  void TearDown(void) {
    EXPECT_NO_THROW(pool->clear());

    delete pool;
  }

  MockCachePool *pool;
};

/**
 * @brief get cache memory
 */
TEST_F(CachePoolTest, get_memory_01_p) {
  EXPECT_CALL(*pool, validate).Times(0);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(1));

  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool->allocate());

  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);

  /* cache addr is invalid until validate() called */
  EXPECT_EQ(mem->getAddr<float>(), nullptr);
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief validate
 */
TEST_F(CachePoolTest, validate_01_p) {
  EXPECT_CALL(*pool, validate).Times(1);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(1));

  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);
  /* cache addr is invalid until validate() called */
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  /* cache addr is valid after validate() called */
  mem->validate();
  EXPECT_NE(mem->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief validate
 */
TEST_F(CachePoolTest, validate_02_p) {
  EXPECT_CALL(*pool, validate).Times(1);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(1));

  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);
  /* cache addr is invalid until validate() called */
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  /* cache addr is valid after validate() called */
  mem->validate();
  EXPECT_NE(mem->getAddr<float>(), nullptr);

  /* double call for validate() has no effect for cache pool validate */
  mem->validate();
  EXPECT_NE(mem->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief invalidate
 */
TEST_F(CachePoolTest, invalidate_01_p) {
  EXPECT_CALL(*pool, validate).Times(1);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(2));

  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);
  /* cache addr is invalid until validate() called */
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  /* cache addr is valid after validate() called */
  mem->validate();
  EXPECT_NE(mem->getAddr<float>(), nullptr);
  /* invalidate() call makes cache addr invalid */
  mem->invalidate();
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief invalidate
 */
TEST_F(CachePoolTest, invalidate_02_p) {
  EXPECT_CALL(*pool, validate).Times(1);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(2));

  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);
  /* cache addr is invalid until validate() called */
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  /* cache addr is valid after validate() called */
  mem->validate();
  EXPECT_NE(mem->getAddr<float>(), nullptr);
  /* cache addr is invalid after invalidate() called */
  mem->invalidate();
  EXPECT_EQ(mem->getAddr<float>(), nullptr);
  /* double call for invalidate() has no effect for cache pool invalidate */
  mem->invalidate();
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief validate & invalidate
 */
TEST_F(CachePoolTest, validate_invalidate_01_p) {
  EXPECT_CALL(*pool, validate).Times(2);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(2));

  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(4, 4, 5);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);
  /* cache addr is invalid until validate() called */
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  /**
   * The data wrote on valid addr is equal to the data after
   * swap out(invalidate) and in(validate).
   */
  mem->validate();
  EXPECT_NE(mem->getAddr<float>(), nullptr);
  *(mem->getAddr<float>()) = TEMP_DATA1;

  mem->invalidate();
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  mem->validate();
  EXPECT_NE(mem->getAddr<float>(), nullptr);
  EXPECT_EQ(*(mem->getAddr<float>()), TEMP_DATA1);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief validate & invalidate
 *        ocuupy same space
 */
TEST_F(CachePoolTest, validate_invalidate_02_n) {
  EXPECT_CALL(*pool, validate).Times(6);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(6));

  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 2);
  auto idx2 = pool->requestMemory(4, 3, 4);
  auto idx3 = pool->requestMemory(4, 5, 6);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::OptimizedV1Planner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 4);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  /**
   * The data wrote on valid addr is equal to the data after
   * swap out(invalidate) and in(validate).
   *
   * Three mems' start-end are not overlaped, so same
   * memory can be used for all.
   */
  mem1->validate();
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  mem2->validate();
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  mem3->validate();
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  *(mem1->getAddr<float>()) = TEMP_DATA1;
  *(mem2->getAddr<float>()) = TEMP_DATA2;
  *(mem3->getAddr<float>()) = TEMP_DATA3;

  mem1->invalidate();
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  mem2->invalidate();
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  mem3->invalidate();
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  mem2->validate();
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  mem3->validate();
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  EXPECT_EQ(*(mem1->getAddr<float>()), TEMP_DATA3);
  EXPECT_EQ(*(mem2->getAddr<float>()), TEMP_DATA3);
  EXPECT_EQ(*(mem3->getAddr<float>()), TEMP_DATA3);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief validate & invalidate
 *        ocuupy different space
 */
TEST_F(CachePoolTest, validate_invalidate_03_p) {
  EXPECT_CALL(*pool, validate).Times(6);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(6));

  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5);
  auto idx2 = pool->requestMemory(4, 3, 8);
  auto idx3 = pool->requestMemory(4, 2, 4);
  EXPECT_NO_THROW(pool->planLayout(nntrainer::OptimizedV1Planner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  /**
   * The data wrote on valid addr is equal to the data after
   * swap out(invalidate) and in(validate).
   *
   * Three mems' start-end are overlaped, so all target memory
   * addr are different.
   */
  mem1->validate();
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  mem2->validate();
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  mem3->validate();
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  *(mem1->getAddr<float>()) = TEMP_DATA1;
  *(mem2->getAddr<float>()) = TEMP_DATA2;
  *(mem3->getAddr<float>()) = TEMP_DATA3;

  mem1->invalidate();
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  mem2->invalidate();
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  mem3->invalidate();
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  mem2->validate();
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  mem3->validate();
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  EXPECT_EQ(*(mem1->getAddr<float>()), TEMP_DATA1);
  EXPECT_EQ(*(mem2->getAddr<float>()), TEMP_DATA2);
  EXPECT_EQ(*(mem3->getAddr<float>()), TEMP_DATA3);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief flush cache data
 */
TEST_F(CachePoolTest, flush_01_p) {
  EXPECT_CALL(*pool, validate).Times(3);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(3));

  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 5, 6, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});
  EXPECT_NO_THROW(pool->planLayout(nntrainer::OptimizedV1Planner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  /**
   * Check loaded data are invalid after flush()
   */
  mem1->validate();
  mem2->validate();
  mem3->validate();
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  pool->flush();
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief load cache data by execution order
 */
TEST_F(CachePoolTest, loadExec_01_p) {
  EXPECT_CALL(*pool, validate).Times(14);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(3));

  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 5, 6, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});
  EXPECT_NO_THROW(pool->planLayout(nntrainer::OptimizedV1Planner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  /**
   * Check loaded data for each execution order
   */
  pool->loadExec(1);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);
  pool->flush();

  pool->loadExec(2);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);
  pool->flush();

  pool->loadExec(3);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);
  pool->flush();

  pool->loadExec(4);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);
  pool->flush();

  pool->loadExec(5);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);
  pool->flush();

  pool->loadExec(6);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);
  pool->flush();

  pool->loadExec(7);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);
  pool->flush();

  pool->loadExec(8);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);
  pool->flush();

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief unload cache data by execution order
 */
TEST_F(CachePoolTest, unloadExec_01_p) {
  EXPECT_CALL(*pool, validate).Times(16);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(16));

  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 5, 6, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});
  EXPECT_NO_THROW(pool->planLayout(nntrainer::OptimizedV1Planner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  /**
   * Check unloaded data for each execution order
   */
  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(1);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(2);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(3);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(4);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(5);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(6);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(7);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  mem1->validate();
  mem2->validate();
  mem3->validate();

  pool->unloadExec(8);
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}

/**
 * @brief load/unload active caches
 */
TEST_F(CachePoolTest, load_unload_actives_01_p) {
  EXPECT_CALL(*pool, validate).Times(3);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(3));

  std::shared_ptr<nntrainer::MemoryData> mem1, mem2, mem3;
  auto idx1 = pool->requestMemory(4, 1, 5, {1, 2, 3, 4, 5});
  auto idx2 = pool->requestMemory(4, 3, 8, {3, 4, 5, 6, 7, 8});
  auto idx3 = pool->requestMemory(4, 2, 4, {2, 3, 4});
  EXPECT_NO_THROW(pool->planLayout(nntrainer::OptimizedV1Planner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 12);
  EXPECT_NO_THROW(mem1 = pool->getMemory(idx1));
  EXPECT_NO_THROW(mem2 = pool->getMemory(idx2));
  EXPECT_NO_THROW(mem3 = pool->getMemory(idx3));
  EXPECT_NE(mem1, nullptr);
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2, nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3, nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  /**
   * Check load and unload acives
   * all active data is invalid when unloadActives() called.
   */
  mem1->validate();
  mem2->validate();
  mem3->validate();

  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  pool->unloadActives();
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  pool->loadActives();
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  pool->unloadActives();
  EXPECT_EQ(mem1->getAddr<float>(), nullptr);
  EXPECT_EQ(mem2->getAddr<float>(), nullptr);
  EXPECT_EQ(mem3->getAddr<float>(), nullptr);

  pool->loadActives();
  EXPECT_NE(mem1->getAddr<float>(), nullptr);
  EXPECT_NE(mem2->getAddr<float>(), nullptr);
  EXPECT_NE(mem3->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
}
