// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_properties.h
 * @date 09 April 2021
 * @brief This file contains test and specification of properties
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <base_properties.h>
#include <util_func.h>

namespace { /**< define a property for testing */

/**
 * @brief banana property tag for example
 *
 */
struct banana_prop_tag : nntrainer::int_prop_tag {};

/**
 * @brief Number of banana property
 *
 */
class NumBanana : public nntrainer::Property<int> {
public:
  NumBanana() { set(1); }                          /**< default value if any */
  static constexpr const char *key = "num_banana"; /**< unique key to access */
  using prop_tag = banana_prop_tag;                /**< property type */

  bool is_valid(const int &v) override { return v >= 0; }
};

/**
 * @brief QualityOfBanana property for example, this has to end with "good"
 *
 */
class QualityOfBanana : public nntrainer::Property<std::string> {
public:
  static constexpr const char *key = "quality_banana";
  using prop_tag = nntrainer::str_prop_tag;

  bool is_valid(const std::string &v) override {
    /// assuming quality of banana property must ends with word "good";
    return nntrainer::endswith(v, "good");
  }
};
} // namespace

TEST(BasicProperty, tagCast) {
  EXPECT_EQ(1, 1); /**< this is to prevent no assert tc from TCM */

  { /**< tag_cast simple cast */
    using T =
      nntrainer::tag_cast<banana_prop_tag, nntrainer::int_prop_tag>::type;
    ::testing::StaticAssertTypeEq<T, nntrainer::int_prop_tag>();
  }

  { /**< tag_cast ranged cast */
    using T = nntrainer::tag_cast<banana_prop_tag, nntrainer::vector_prop_tag,
                                  nntrainer::int_prop_tag>::type;
    ::testing::StaticAssertTypeEq<T, nntrainer::int_prop_tag>();
  }

  { /**< tag_cast does not have appropriate candidates */
    using T = nntrainer::tag_cast<banana_prop_tag, int, std::string>::type;
    ::testing::StaticAssertTypeEq<T, banana_prop_tag>();
  }
}

TEST(BasicProperty, valid_p) {
  { /** set -> get / to_string */
    NumBanana b;
    b.set(123);
    EXPECT_EQ(b.get(), 123);
    EXPECT_EQ(nntrainer::to_string(b), "123");

    QualityOfBanana q;
    q.set("this is good");
    EXPECT_EQ(q.get(), "this is good");
    EXPECT_EQ(nntrainer::to_string(q), "this is good");
  }

  { /**< from_string -> get / to_string */
    NumBanana b;
    nntrainer::from_string("3", b);
    EXPECT_EQ(b.get(), 3);
    EXPECT_EQ(nntrainer::to_string(b), "3");

    QualityOfBanana q;
    nntrainer::from_string("this is good", q);
    EXPECT_EQ(q.get(), "this is good");
    EXPECT_EQ(nntrainer::to_string(q), "this is good");
  }
}

TEST(BasicProperty, setNotValid_01_n) {
  NumBanana b;
  EXPECT_THROW(b.set(-1), std::invalid_argument);
}

TEST(BasicProperty, setNotValid_02_n) {
  QualityOfBanana q;
  EXPECT_THROW(q.set("invalid_str"), std::invalid_argument);
}

TEST(BasicProperty, fromStringNotValid_01_n) {
  NumBanana b;
  EXPECT_THROW(nntrainer::from_string("not integer", b), std::invalid_argument);
}

TEST(BasicProperty, fromStringNotValid_02_n) {
  NumBanana b;
  EXPECT_THROW(nntrainer::from_string("-1", b), std::invalid_argument);
}

TEST(BasicProperty, fromStringNotValid_03_n) {
  QualityOfBanana q;
  EXPECT_THROW(nntrainer::from_string("invalid_str", q), std::invalid_argument);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
