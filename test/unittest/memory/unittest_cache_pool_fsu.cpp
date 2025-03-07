// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_cache_pool_fsu.cpp
 * @date 05 Mar 2025
 * @brief Cache Pool Test for FSU
 * @see https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <basic_planner.h>
#include <cache_pool.h>
#include <fstream>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optimized_v3_planner.h>

/**
 * @brief Mock class for cache pool
 */
class MockCachePoolFSU : public nntrainer::CachePool {
public:
  MockCachePoolFSU(const std::string &file_path, const std::string &name,
                   ml::train::ExecutionMode exec_mode) :
    nntrainer::CachePool(file_path, name, exec_mode) {
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
class CachePoolFSUTest : public ::testing::Test {
public:
  void SetUp(void) {
    pool = new MockCachePoolFSU("", "weight.bin",
                                ml::train::ExecutionMode::INFERENCE);
  }

  void TearDown(void) {
    EXPECT_NO_THROW(pool->clear());
    delete pool;
  }

  MockCachePoolFSU *pool;
};

void MakeWightFile(size_t size) {
  std::ofstream outFile("weight.bin", std::ios::out | std::ios::binary);
  char *random_data = static_cast<char *>(calloc(size, 1));
  for (size_t i = 0; i < size; i++) {
    random_data[i] = 0xAA;
  }
  outFile.write(reinterpret_cast<const char *>(random_data), size);
  free(random_data);
  outFile.close();
}

void RemoveWeightFile() { remove("./weight.bin"); }

/**
 * @brief Check CachePool init for FSU
 */
TEST_F(CachePoolFSUTest, check_init) {
  EXPECT_EQ("weight.bin", pool->getName());
  EXPECT_EQ(ml::train::ExecutionMode::INFERENCE, pool->getExecMode());
}

/**
 * @brief CachePool FSU allocate Test
 */
TEST_F(CachePoolFSUTest, check_allocate) {
  MakeWightFile(4096);
  EXPECT_CALL(*pool, validate).Times(0);
  EXPECT_CALL(*pool, invalidate).Times(testing::AtLeast(1));

  std::shared_ptr<nntrainer::MemoryData> mem;
  auto idx = pool->requestMemory(4096, 0, 2, {1, 2, 3});
  EXPECT_NO_THROW(pool->planLayout(nntrainer::OptimizedV3Planner()));
  EXPECT_NO_THROW(pool->allocate());
  EXPECT_EQ(pool->size(), 4096);
  EXPECT_NO_THROW(mem = pool->getMemory(idx));
  EXPECT_NE(mem, nullptr);

  /* cache addr is invalid until validate() called */
  EXPECT_EQ(mem->getAddr<float>(), nullptr);
  EXPECT_EQ(mem->getAddr<float>(), nullptr);

  EXPECT_NO_THROW(pool->deallocate());
  RemoveWeightFile();
}
