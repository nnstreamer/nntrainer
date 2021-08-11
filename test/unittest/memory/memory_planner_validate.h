// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_planner_validate.h
 * @date   11 August 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 memory_planner_validate.h
 */

#ifndef __MEMORY_PLANNER_VALIDATE_H__
#define __MEMORY_PLANNER_VALIDATE_H__

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include <memory_planner.h>

/**
 * @class MemoryPlannerValidate
 * @brief Memory planning validation
 */
class MemoryPlannerValidate : public ::testing::TestWithParam<std::string> {
public:
  /**
   * @brief Destructor
   *
   */
  virtual ~MemoryPlannerValidate() {}

  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp();

  /**
   * @brief Release test resources
   *
   */
  virtual void TearDown() {}

protected:
  std::unique_ptr<nntrainer::MemoryPlanner> planner;
  std::string plan_type;
};

#endif // __MEMORY_PLANNER_VALIDATE_H__
