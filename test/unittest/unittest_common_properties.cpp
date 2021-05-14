// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_common_properties.h
 * @date 15 May 2021
 * @brief This file contains test and specification of properties and exporter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <common_properties.h>

/// @todo change this to typed param test
/// <type, list of string, value pair, list of invalid string, value pair>
TEST(NameProperty, setPropertyValid_p) {
  nntrainer::props::Name n;
  EXPECT_NO_THROW(n.set("layer"));
  EXPECT_EQ(n.get(), "layer");

  EXPECT_NO_THROW(n.set("layer-"));
  EXPECT_EQ(n.get(), "layer-");

  EXPECT_NO_THROW(n.set("laye-r"));
  EXPECT_EQ(n.get(), "laye-r");

  EXPECT_NO_THROW(n.set("layer/a"));
  EXPECT_EQ(n.get(), "layer/a");

  EXPECT_NO_THROW(n.set("laye__r"));
  EXPECT_EQ(n.get(), "laye__r");
}

TEST(NameProperty, forbiddenString_01_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("layer "), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_02_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("layer layer"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_03_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set(" layer"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_04_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("layer,"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_05_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("lay,er"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_06_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("lay, er"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_07_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set(",layer"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_08_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("layer+"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_09_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("la+ yer"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_10_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("lay+er"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_11_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("+layer"), std::invalid_argument);
}

TEST(NameProperty, forbiddenString_12_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("+ layer"), std::invalid_argument);
}

TEST(NameProperty, mustStartWithAlphaNumeric_01_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("+layer"), std::invalid_argument);
}

TEST(InputSpecProperty, setPropertyValid_p) {
  using namespace nntrainer::props;
  {
    InputSpec expected(
      ConnectionSpec({Name("A"), Name("B"), Name("C")}, "concat"));

    InputSpec actual;
    nntrainer::from_string("A, B, C", actual);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ("A,B,C", nntrainer::to_string(actual));
  }

  {
    InputSpec expected(
      ConnectionSpec({Name("A"), Name("B"), Name("C")}, "addition"));

    InputSpec actual;
    nntrainer::from_string("A+ B +C", actual);
    EXPECT_EQ("A+B+C", nntrainer::to_string(actual));

    EXPECT_EQ(actual, expected);
  }

  {
    InputSpec expected(ConnectionSpec({Name("A")}, ConnectionSpec::NoneType));

    InputSpec actual;
    nntrainer::from_string("A", actual);
    EXPECT_EQ("A", nntrainer::to_string(actual));

    EXPECT_EQ(actual, expected);
  }
}

TEST(InputSpecProperty, emptyString_n_01) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("", actual), std::invalid_argument);
}

TEST(InputSpecProperty, combinedOperator_n_01) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("A,B+C", actual), std::invalid_argument);
}

TEST(InputSpecProperty, combinedOperator_n_02) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("A+B,C", actual), std::invalid_argument);
}

TEST(InputSpecProperty, noOperator_n_01) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("A B", actual), std::invalid_argument);
}

TEST(InputSpecProperty, noOperator_n_02) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("A B", actual), std::invalid_argument);
}

TEST(InputSpecProperty, leadingOperator_n_01) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string(",A,B", actual), std::invalid_argument);
}

TEST(InputSpecProperty, leadingOperator_n_02) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("+A+B", actual), std::invalid_argument);
}

TEST(InputSpecProperty, trailingOperator_n_01) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("A,B,,", actual), std::invalid_argument);
}

TEST(InputSpecProperty, trailingOperator_n_02) {
  using namespace nntrainer::props;
  InputSpec actual;
  EXPECT_THROW(nntrainer::from_string("A+B++", actual), std::invalid_argument);
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
