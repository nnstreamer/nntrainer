// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_planner_validate.h
 * @date   11 August 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Tests for memory planning
 */

#include <bitset>
#include <random>
#include <vector>

#include <basic_planner.h>
#include <memory_planner_validate.h>

constexpr unsigned int MEM_BYTES = 50;
constexpr unsigned int MEM_QUANT = 100;
constexpr unsigned int INTERVAL_SIZE = 5;

void MemoryPlannerValidate::SetUp() {
  plan_type = GetParam();

  planner = nullptr;
  if (plan_type == nntrainer::BasicPlanner::type)
    planner = std::make_unique<nntrainer::BasicPlanner>();
  else
    throw std::invalid_argument("Invalid planner type");
}

/**
 * @brief Validate the provided layout does not overlap
 */
static bool validateNoOverlap(const std::vector<size_t> &memory_size,
                              const std::vector<size_t> &memory_offset,
                              unsigned int size) {
  std::vector<bool> mem_overlap(size, false);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    for (unsigned int mem = memory_offset[idx];
         mem < memory_offset[idx] + memory_size[idx]; mem++) {
      EXPECT_FALSE(mem_overlap[mem]);
      if (mem_overlap[mem])
        return false;
      mem_overlap[mem] = true;
    }
  }

  /** This ensures that there are no unnecessary gaps in between */
  EXPECT_EQ(mem_overlap.size(),
            std::accumulate(mem_overlap.begin(), mem_overlap.end(), 0u));

  return true;
}

/**
 * @brief Validate the provided layout does completely overlap
 */
static bool validateAllOverlap(const std::vector<size_t> &memory_size,
                               const std::vector<size_t> &memory_offset) {
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    EXPECT_EQ(memory_offset[idx], memory_offset[0]);
  }

  return true;
}

/**
 * @brief Validate the provided layout does not overflow outside the given
 * size
 */
static bool validateOverflow(const std::vector<size_t> &memory_size,
                             const std::vector<size_t> &memory_offset,
                             size_t pool_size) {
  for (unsigned int idx = 0; idx < memory_size.size(); idx++)
    if (memory_offset[idx] + memory_size[idx] > pool_size)
      return false;

  return true;
}

/**
 * @brief Test memory planning with full overlap of requested memories
 * @details validityIsFullOverlapping - validify is the full overlapping memory
 * requests handled correctly.
 */
TEST_P(MemoryPlannerValidate, full_overlap) {
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);

  std::vector<size_t> memory_size(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    memory_size[idx] = dist(rng);
  }
  std::vector<std::pair<unsigned int, unsigned int>> memory_validity(MEM_QUANT,
                                                                     {1, 2});
  std::vector<size_t> memory_offset;

  size_t pool_size =
    planner->planLayout(memory_size, memory_validity, memory_offset);

  EXPECT_EQ(pool_size,
            std::accumulate(memory_size.begin(), memory_size.end(), 0u));
  EXPECT_TRUE(validateNoOverlap(memory_size, memory_offset, pool_size));
  EXPECT_TRUE(validateOverflow(memory_size, memory_offset, pool_size));
}

/**
 * @brief Test memory planning with no overlap of requested memories
 */
TEST_P(MemoryPlannerValidate, none_overlap) {
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);

  std::vector<size_t> memory_size(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    memory_size[idx] = dist(rng);
  }

  std::vector<std::pair<unsigned int, unsigned int>> memory_validity(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_validity.size(); idx++) {
    memory_validity[idx] = {idx, idx + 1};
  }

  std::vector<size_t> memory_offset;
  size_t pool_size =
    planner->planLayout(memory_size, memory_validity, memory_offset);

  EXPECT_TRUE(validateOverflow(memory_size, memory_offset, pool_size));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool_size,
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
    EXPECT_TRUE(validateNoOverlap(memory_size, memory_offset, pool_size));
  } else {
    EXPECT_EQ(pool_size,
              *std::max_element(memory_size.begin(), memory_size.end()));
    EXPECT_TRUE(validateAllOverlap(memory_size, memory_offset));
  }
}

/**
 * @brief Test memory planning with partial overlap of requested memories
 */
TEST_P(MemoryPlannerValidate, partial_overlap) {
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> dist(1, MEM_BYTES);
  std::uniform_int_distribution<unsigned int> dist_interval(1, INTERVAL_SIZE);

  std::vector<size_t> memory_size(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    memory_size[idx] = dist(rng);
  }

  std::vector<std::pair<unsigned int, unsigned int>> memory_validity(MEM_QUANT);
  for (unsigned int idx = 0; idx < memory_validity.size(); idx++) {
    memory_validity[idx] = {idx, idx + dist_interval(rng)};
  }
  std::shuffle(memory_validity.begin(), memory_validity.end(),
               std::default_random_engine(0));

  std::vector<size_t> memory_offset;
  size_t pool_size =
    planner->planLayout(memory_size, memory_validity, memory_offset);

  EXPECT_TRUE(validateOverflow(memory_size, memory_offset, pool_size));
  if (planner->getType() == nntrainer::BasicPlanner::type) {
    EXPECT_EQ(pool_size,
              std::accumulate(memory_size.begin(), memory_size.end(), 0u));
    EXPECT_TRUE(validateNoOverlap(memory_size, memory_offset, pool_size));
  } else {
    EXPECT_EQ(pool_size,
              *std::max_element(memory_size.begin(), memory_size.end()));
    // EXPECT_TRUE(validateIntervalOverlap(memory_size, memory_offset));
  }
}
