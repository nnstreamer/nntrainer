// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_compiler.cpp
 * @date 01 April 2021
 * @brief compiler test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <compiler.h>

TEST(scaffoldingTest, sampleTest) { EXPECT_NO_THROW(nntrainer::hello_world()); }
