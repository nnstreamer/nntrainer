// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_memory_pool.cpp
 * @date 11 August 2021
 * @brief Memory Pool Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <basic_planner.h>
#include <memory_planner_validate.h>
#include <memory_pool.h>

constexpr unsigned int MEM_BYTES = 128;
constexpr unsigned int MEM_QUANT = 100;
constexpr unsigned int INTERVAL_SIZE = 5;

/**
 * @brief creation and destruction
 */
TEST(MemoryPool, create_destroy) { EXPECT_NO_THROW(nntrainer::MemoryPool()); }

/**
 * @brief request 0 sized memory
 */
TEST(MemoryPool, request_mem_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.requestMemory(0, 1, 2), std::invalid_argument);
}

/**
 * @brief request memory when starts after it ends
 */
TEST(MemoryPool, request_mem_02_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.requestMemory(1, 3, 2), std::invalid_argument);
}

/**
 * @brief request memory with 0 valid time
 */
TEST(MemoryPool, request_mem_03_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.requestMemory(1, 4, 4), std::invalid_argument);
}

/**
 * @brief request memory after allocate
 */
TEST(MemoryPool, request_mem_04_n) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.requestMemory(1, 5, 6), std::invalid_argument);
}

/**
 * @brief request memory
 */
TEST(MemoryPool, request_mem_05_p) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
}

/**
 * @brief plan layout without reqeustMemory
 */
TEST(MemoryPool, plan_layout_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.planLayout(nntrainer::BasicPlanner()), std::runtime_error);
}

/**
 * @brief plan layout after allocate
 */
TEST(MemoryPool, plan_layout_02_n) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.planLayout(nntrainer::BasicPlanner()), std::runtime_error);
}

/**
 * @brief plan layout
 */
TEST(MemoryPool, plan_layout_03_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_EQ(1u, pool.size());

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_EQ(2u, pool.size());
}

/**
 * @brief deallocate
 */
TEST(MemoryPool, deallocate_01_p) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.deallocate());

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief allocate without requestMemory
 */
TEST(MemoryPool, allocate_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.allocate(), std::runtime_error);
}

/**
 * @brief allocate without planLayout
 */
TEST(MemoryPool, allocate_02_n) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_THROW(pool.allocate(), std::runtime_error);
}

/**
 * @brief allocate aftrer allocate
 */
TEST(MemoryPool, allocate_03_n) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.allocate(), std::runtime_error);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief allocate
 */
TEST(MemoryPool, allocate_04_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(3, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_EQ(3u, pool.size());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());

  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief size of the pool
 */
TEST(MemoryPool, size_01_p) {
  nntrainer::MemoryPool pool;

  EXPECT_EQ(pool.size(), 0u);

  pool.requestMemory(1, 4, 5);
  EXPECT_EQ(pool.size(), 0u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.size(), 1u);

  pool.allocate();
  EXPECT_EQ(pool.size(), 1u);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief min requirement
 */
TEST(MemoryPool, min_mem_req_01_p) {
  nntrainer::MemoryPool pool;

  EXPECT_EQ(pool.minMemoryRequirement(), 0u);

  pool.requestMemory(1, 4, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** exact overlap */
  pool.requestMemory(2, 4, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap */
  pool.requestMemory(3, 2, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);

  /** start overlap */
  pool.requestMemory(4, 4, 8);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 10u);

  /** complete overlap */
  pool.requestMemory(5, 1, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 15u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 15u);
}

/**
 * @brief min requirement
 */
TEST(MemoryPool, min_mem_req_02_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 5, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** partial overlap */
  pool.requestMemory(2, 1, 8);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap */
  pool.requestMemory(3, 7, 12);
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 6u);
}

/**
 * @brief min requirement
 */
TEST(MemoryPool, min_mem_req_03_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 5, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** partial overlap */
  pool.requestMemory(2, 1, 8);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap with matching ends */
  pool.requestMemory(3, 8, 12);
  EXPECT_EQ(pool.minMemoryRequirement(), 4u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 4u);
}

/**
 * @brief min requirement
 */
TEST(MemoryPool, min_mem_req_04_p) {
  nntrainer::MemoryPool pool;

  pool.requestMemory(1, 5, 10);
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 1u);

  /** partial overlap */
  pool.requestMemory(2, 1, 5);
  EXPECT_EQ(pool.minMemoryRequirement(), 2u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 2u);

  /** ending overlap with matching ends */
  pool.requestMemory(3, 10, 12);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  /** ending overlap with matching ends */
  pool.requestMemory(1, 12, 13);
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);

  pool.planLayout(nntrainer::BasicPlanner());
  EXPECT_EQ(pool.minMemoryRequirement(), 3u);
}

/**
 * @brief get memory
 */
TEST(MemoryPool, get_memory_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_THROW(pool.getMemory(1), std::invalid_argument);
}

/**
 * @brief get memory
 */
TEST(MemoryPool, get_memory_02_n) {
  nntrainer::MemoryPool pool;

  auto idx = pool.requestMemory(1, 4, 5);
  EXPECT_THROW(pool.getMemory(idx), std::invalid_argument);
}

/**
 * @brief get memory
 */
TEST(MemoryPool, get_memory_03_n) {
  nntrainer::MemoryPool pool;

  auto idx = pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_ANY_THROW(pool.getMemory(idx + 1));

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief get memory
 */
TEST(MemoryPool, get_memory_04_p) {
  nntrainer::MemoryPool pool;
  std::shared_ptr<nntrainer::MemoryData<float>> mem;

  auto idx = pool.requestMemory(1, 4, 5);
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_NO_THROW(mem = pool.getMemory(idx));
  EXPECT_NE(mem, nullptr);

  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief clear after allocate
 */
TEST(MemoryPool, clear_01_n) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));
  EXPECT_NO_THROW(pool.allocate());

  EXPECT_THROW(pool.clear(), std::invalid_argument);
  EXPECT_NO_THROW(pool.deallocate());
}

/**
 * @brief clear
 */
TEST(MemoryPool, clear_02_p) {
  nntrainer::MemoryPool pool;

  EXPECT_NO_THROW(pool.requestMemory(1, 4, 5));
  EXPECT_NO_THROW(pool.planLayout(nntrainer::BasicPlanner()));

  EXPECT_NO_THROW(pool.clear());
}

/**
 * @brief validate memory full overlap
 */
TEST_P(MemoryPlannerValidate, validate_memory_full_overlap) {
  nntrainer::MemoryPool pool;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);

  std::vector<unsigned int> tokens(MEM_QUANT);
  std::vector<size_t> memory_size(MEM_QUANT);
  std::vector<std::shared_ptr<nntrainer::MemoryData<float>>> ptrs(MEM_QUANT);

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    memory_size[idx] = dist(rng);
    unsigned int start = 1;
    unsigned int end = start + dist_interval(rng);
    EXPECT_NO_THROW(tokens[idx] =
                      pool.requestMemory(memory_size[idx], start, end));
  }

  EXPECT_NO_THROW(pool.planLayout(*planner.get()));
  EXPECT_EQ(pool.size(),
            std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  EXPECT_NO_THROW(pool.allocate());

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    EXPECT_NO_THROW(ptrs[idx] = pool.getMemory(tokens[idx]));

  /** write data to memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    memset(ptrs[idx]->getAddr(), idx, memory_size[idx]);

  /** verify data in memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    std::vector<unsigned char> golden(memory_size[idx], idx);
    memcmp(ptrs[idx]->getAddr(), &golden[0], memory_size[idx]);
  }

  pool.deallocate();
}

/**
 * @brief validate memory no overlap
 */
TEST_P(MemoryPlannerValidate, validate_memory_no_overlap) {
  nntrainer::MemoryPool pool;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);

  std::vector<unsigned int> tokens(MEM_QUANT);
  std::vector<size_t> memory_size(MEM_QUANT);
  std::vector<std::shared_ptr<nntrainer::MemoryData<float>>> ptrs(MEM_QUANT);

  unsigned int prev_idx = 0;
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    memory_size[idx] = dist(rng);
    unsigned int start = prev_idx;
    unsigned int end = start + dist_interval(rng);
    EXPECT_NO_THROW(tokens[idx] =
                      pool.requestMemory(memory_size[idx], start, end));
    prev_idx = end;
  }

  EXPECT_NO_THROW(pool.planLayout(*planner.get()));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool.size(),
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  } else {
    EXPECT_EQ(pool.size(),
              *std::max_element(memory_size.begin(), memory_size.end()));
  }
  EXPECT_NO_THROW(pool.allocate());

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    EXPECT_NO_THROW(ptrs[idx] = pool.getMemory(tokens[idx]));

  /** write data to memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    memset(ptrs[idx]->getAddr(), idx, memory_size[idx]);

  /** verify data in memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    std::vector<unsigned char> golden(memory_size[idx], idx);
    memcmp(ptrs[idx]->getAddr(), &golden[0], memory_size[idx]);
  }

  pool.deallocate();
}

/**
 * @brief validate memory partial overlap
 */
TEST_P(MemoryPlannerValidate, validate_memory_partial_overlap) {
  nntrainer::MemoryPool pool;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);
  std::uniform_int_distribution<unsigned int> dist_interval_start(1, 100);

  std::vector<unsigned int> tokens(MEM_QUANT);
  std::vector<size_t> memory_size(MEM_QUANT);
  std::vector<std::shared_ptr<nntrainer::MemoryData<float>>> ptrs(MEM_QUANT);

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    memory_size[idx] = dist(rng);
    unsigned int start = dist_interval_start(rng);
    unsigned int end = start + dist_interval(rng);
    EXPECT_NO_THROW(tokens[idx] =
                      pool.requestMemory(memory_size[idx], start, end));
  }

  EXPECT_NO_THROW(pool.planLayout(*planner.get()));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool.size(),
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  } else {
    EXPECT_GE(pool.size(),
              *std::max_element(memory_size.begin(), memory_size.end()));
    EXPECT_LE(pool.size(),
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  }
  EXPECT_NO_THROW(pool.allocate());

  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    EXPECT_NO_THROW(ptrs[idx] = pool.getMemory(tokens[idx]));

  /** write data to memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++)
    memset(ptrs[idx]->getAddr(), idx, memory_size[idx]);

  /** verify data in memory */
  for (unsigned int idx = 0; idx < MEM_QUANT; idx++) {
    std::vector<unsigned char> golden(memory_size[idx], idx);
    memcmp(ptrs[idx]->getAddr(), &golden[0], memory_size[idx]);
  }

  pool.deallocate();
}
