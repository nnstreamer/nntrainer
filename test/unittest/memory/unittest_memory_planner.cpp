// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_memory_planning.cpp
 * @date 11 August 2021
 * @brief Memory Planner Test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory_planner_validate.h>

#include <basic_planner.h>

INSTANTIATE_TEST_CASE_P(BasicPlanner, MemoryPlannerValidate,
                        ::testing::Values(nntrainer::BasicPlanner::type));
