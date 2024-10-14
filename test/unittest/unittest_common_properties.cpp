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
#include <connection.h>
#include <nntrainer_test_util.h>
#include <tensor_dim.h>

#include <array>
/**
 * @brief NamePropertyTest
 * @tparam std::string string which will be used as name
 */
class NamePropertyTest : public ::testing::TestWithParam<std::string> {};

TEST_P(NamePropertyTest, setPropertyValid_p) {
  nntrainer::props::Name n;
  std::string param = GetParam();

  EXPECT_NO_THROW(n.set(param));
  EXPECT_EQ(n.get(), param);
}

GTEST_PARAMETER_TEST(NamePropertyTests, NamePropertyTest,
                     ::testing::Values("layer", "layer-", "laye-r", "layer/a",
                                       "laye__r"));

/**
 * @brief NameTest
 * @tparam std::string string which will be added as suffix to name
 */
class NameTest : public ::testing::TestWithParam<std::string> {
public:
  ~NameTest() {}

  /**
   * @brief SetUp test cases here
   *
   */
  virtual void SetUp() { suffix = GetParam(); }

  /**
   * @brief do here if any memory needs to be released
   *
   */
  virtual void TearDown() {}

protected:
  std::string suffix;
};

TEST_P(NameTest, forbiddenSuffix_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("name" + suffix), std::invalid_argument);
}

GTEST_PARAMETER_TEST(ForbiddenSuffixTests, NameTest,
                     ::testing::Values("!", "@", "#", "$", "%", "^", "&", "*",
                                       "=", "+0", "(0)", "{0}", "[0]", "<0>",
                                       ";", ":", ",", "?", " ", " layer"));

TEST(NamePropertyTest, mustStartWithAlphaNumeric_01_n) {
  nntrainer::props::Name n;
  EXPECT_THROW(n.set("/layer"), std::invalid_argument);
}

TEST(InputConnection, setPropertyValid_p) {
  using namespace nntrainer::props;
  {
    InputConnection expected(nntrainer::Connection("a", 0));

    InputConnection actual;
    nntrainer::from_string("A", actual);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ("a(0)", nntrainer::to_string(actual));
  }

  {
    InputConnection expected(nntrainer::Connection("a", 2));

    InputConnection actual;
    nntrainer::from_string("a(2)", actual);
    EXPECT_EQ(actual, expected);
    EXPECT_EQ("a(2)", nntrainer::to_string(actual));
  }
}

TEST(InputConnection, emptyString_n_01) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("", actual), std::invalid_argument);
}

TEST(InputConnection, onlyIndex_n_01) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("(0)", actual), std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_01) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("a(0", actual), std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_02) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("(0", actual), std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_03) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("a((0))", actual), std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_04) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("a((0)", actual), std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_05) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("a(0))", actual), std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_06) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("a(0)(1)", actual),
               std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_07) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("name:layer0", actual),
               std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_08) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("name(layer0)", actual),
               std::invalid_argument);
}

TEST(InputConnection, invalidFormat_n_09) {
  using namespace nntrainer::props;
  InputConnection actual;
  EXPECT_THROW(nntrainer::from_string("name==layer0", actual),
               std::invalid_argument);
}

TEST(DropOutRate, dropout_01_n) {
  nntrainer::props::DropOutRate dropout;
  EXPECT_THROW(dropout.set(-0.5), std::invalid_argument);
}

TEST(NumClass, numclass_01_n) {
  nntrainer::props::NumClass numclass;
  EXPECT_THROW(numclass.set(0), std::invalid_argument);
}

TEST(Momentum, momentum_01_n) {
  nntrainer::props::Momentum momentum;
  EXPECT_THROW(momentum.set(0), std::invalid_argument);
}

TEST(Momentum, momentum_02_n) {
  nntrainer::props::Momentum momentum;
  EXPECT_THROW(momentum.set(1), std::invalid_argument);
}

TEST(SplitDimension, split_dimension_01_n) {
  nntrainer::props::SplitDimension split_dimension;
  EXPECT_THROW(split_dimension.set(0), std::invalid_argument);
}

TEST(Padding2D, setPropertyValid_p) {
  nntrainer::props::Padding2D p;
  EXPECT_NO_THROW(p.set("same"));
  EXPECT_EQ(p.get(), "same");

  EXPECT_NO_THROW(p.set("Same"));
  EXPECT_EQ(p.get(), "Same");

  EXPECT_EQ(p.compute({32, 32}, {3, 3}, {1, 1}, {1, 1}),
            (std::array<unsigned int, 4>({1, 1, 1, 1})));

  EXPECT_NO_THROW(p.set("valid"));
  EXPECT_EQ(p.get(), "valid");

  EXPECT_EQ(p.compute({32, 32}, {3, 3}, {1, 1}, {1, 1}),
            (std::array<unsigned int, 4>({0, 0, 0, 0})));

  EXPECT_NO_THROW(p.set("1"));
  EXPECT_EQ(p.get(), "1");
  EXPECT_EQ(p.compute({32, 32}, {3, 3}, {1, 1}, {1, 1}),
            (std::array<unsigned int, 4>({1, 1, 1, 1})));

  EXPECT_NO_THROW(p.set("0"));
  EXPECT_EQ(p.get(), "0");

  EXPECT_NO_THROW(p.set("1, 2"));
  EXPECT_EQ(p.get(), "1, 2");
  EXPECT_EQ(p.compute({32, 32}, {3, 3}, {1, 1}, {1, 1}),
            (std::array<unsigned int, 4>({1, 1, 2, 2})));

  EXPECT_NO_THROW(p.set("1, 2, 3, 4"));
  EXPECT_EQ(p.get(), "1, 2, 3, 4");
  EXPECT_EQ(p.compute({32, 32}, {3, 3}, {1, 1}, {1, 1}),
            (std::array<unsigned int, 4>({1, 2, 3, 4})));
}

TEST(Padding2D, randomString_01_n) {
  nntrainer::props::Padding2D p;
  EXPECT_THROW(p.set("seme"), std::invalid_argument);
}

TEST(Padding2D, randomString_02_n) {
  nntrainer::props::Padding2D p;
  EXPECT_THROW(p.set("velid"), std::invalid_argument);
}

TEST(Padding2D, given_padding_of_three_n) {
  nntrainer::props::Padding2D p;
  EXPECT_THROW(p.set("1, 2, 3"), std::invalid_argument);
}

TEST(Padding2D, given_padding_is_negative_01_n) {
  nntrainer::props::Padding2D p;
  EXPECT_THROW(p.set("-1"), std::invalid_argument);
}

TEST(Padding2D, given_padding_is_negative_02_n) {
  nntrainer::props::Padding2D p;
  EXPECT_THROW(p.set("-1, 1"), std::invalid_argument);
}

TEST(BasicRegularizerConstant, basic_regularizer_constant_01_n) {
  nntrainer::props::BasicRegularizerConstant basic_regularizer_constant;
  EXPECT_THROW(basic_regularizer_constant.set(-1), std::invalid_argument);
}

TEST(BasicRegularizer, basic_regularizer_01_n) {
  EXPECT_THROW(nntrainer::props::BasicRegularizer basic_regularizer(
                 nntrainer::WeightRegularizer::UNKNOWN),
               std::invalid_argument);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during IniGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  return result;
}
