// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file	simpleshot_utils_test.cpp
 * @date	08 Jan 2021
 * @brief	test for simpleshot utils
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <gtest/gtest.h>
#include <simpleshot_utils.h>

namespace simpleshot {
TEST(getKeyValue, parse_okay_p) {
  {
    util::Entry e = util::getKeyValue("abc=123");
    EXPECT_EQ(e.key, "abc");
    EXPECT_EQ(e.value, "123");
  }
  {
    util::Entry e = util::getKeyValue("abc = 123");
    EXPECT_EQ(e.key, "abc");
    EXPECT_EQ(e.value, "123");
  }
  {
    util::Entry e = util::getKeyValue("abc  = 123");
    EXPECT_EQ(e.key, "abc");
    EXPECT_EQ(e.value, "123");
  }
  {
    util::Entry e = util::getKeyValue("abc =  123");
    EXPECT_EQ(e.key, "abc");
    EXPECT_EQ(e.value, "123");
  }
}

TEST(getKeyValue, invalid_format_01_n) {
  EXPECT_THROW(util::getKeyValue("abc"), std::invalid_argument);
}

TEST(getKeyValue, invalid_format_02_n) {
  EXPECT_THROW(util::getKeyValue("abc="), std::invalid_argument);
}

TEST(getKeyValue, invalid_format_03_n) {
  EXPECT_THROW(util::getKeyValue("abc=1=2"), std::invalid_argument);
}

TEST(getKeyValue, invalid_format_04_n) {
  EXPECT_THROW(util::getKeyValue("=12"), std::invalid_argument);
}
} // namespace simpleshot
