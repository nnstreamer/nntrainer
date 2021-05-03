// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_properties.h
 * @date 09 April 2021
 * @brief This file contains test and specification of properties and exporter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <utility>

#include <base_properties.h>
#include <fc_layer.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
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

/**
 * @brief BananaTypes property for example
 *
 */
class BananaTypes : public nntrainer::Property<std::vector<unsigned int>> {
public:
  static constexpr const char *key = "banana_types";
  using prop_tag = nntrainer::vector_prop_tag;

  bool is_valid(const std::vector<unsigned int> &v) override {
    return v.size() > 3;
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

/// @todo convert this to typed param test
TEST(BasicProperty, valid_p) {
  { /** set -> get / to_string, int*/
    NumBanana b;
    b.set(123);
    EXPECT_EQ(b.get(), 123);
    EXPECT_EQ(nntrainer::to_string(b), "123");
  }

  { /**< from_string -> get / to_string, int*/
    NumBanana b;
    nntrainer::from_string("3", b);
    EXPECT_EQ(b.get(), 3);
    EXPECT_EQ(nntrainer::to_string(b), "3");
  }

  { /** set -> get / to_string, string*/
    QualityOfBanana q;
    q.set("this is good");
    EXPECT_EQ(q.get(), "this is good");
    EXPECT_EQ(nntrainer::to_string(q), "this is good");
  }

  { /**< from_string -> get / to_string, string prop */
    QualityOfBanana q;
    nntrainer::from_string("this is good", q);
    EXPECT_EQ(q.get(), "this is good");
    EXPECT_EQ(nntrainer::to_string(q), "this is good");
  }

  { /** set -> get / to_string, uint vector prop*/
    BananaTypes q;
    q.set({1, 2, 3, 4});
    EXPECT_EQ(q.get(), std::vector<unsigned int>({1, 2, 3, 4}));
    EXPECT_EQ(nntrainer::to_string(q), "1,2,3,4");
  }

  { /**< from_string -> get / to_string, uint vector prop */
    BananaTypes q;
    nntrainer::from_string("1, 2,3, 4, 5", q);
    EXPECT_EQ(q.get(), std::vector<unsigned int>({1, 2, 3, 4, 5}));
    EXPECT_EQ(nntrainer::to_string(q), "1,2,3,4,5");
  }

  { /**< exporter test */
    auto props = std::make_tuple(NumBanana(), QualityOfBanana());

    nntrainer::Exporter e;
    e.save_result(props, nntrainer::ExportMethods::METHOD_STRINGVECTOR);

    auto result = e.get_result<nntrainer::ExportMethods::METHOD_STRINGVECTOR>();

    auto pair = std::pair<std::string, std::string>("num_banana", "1");
    EXPECT_EQ(result[0], pair);

    auto pair2 = std::pair<std::string, std::string>("quality_banana", "");
    EXPECT_EQ(result[1], pair2);
  }

  { /**< export from layer */
    auto layer = nntrainer::FullyConnectedLayer(1);
    nntrainer::Exporter e;
    layer.export_to(e);

    auto result = e.get_result<nntrainer::ExportMethods::METHOD_STRINGVECTOR>();
    auto pair0 = std::pair<std::string, std::string>("name", "");
    EXPECT_EQ(result[0], pair0);
    auto pair1 = std::pair<std::string, std::string>("unit", "1");
    EXPECT_EQ(result[1], pair1);
  }

  { /**< load from layer */
    auto props = std::make_tuple(NumBanana(), QualityOfBanana());

    auto v =
      nntrainer::loadProperties({"num_banana=2", "quality_banana=thisisgood",
                                 "num_banana=42", "not_used=key"},
                                props);

    EXPECT_EQ(v, std::vector<std::string>{"not_used=key"});
    EXPECT_EQ(std::get<0>(props).get(), 42);
    EXPECT_EQ(std::get<1>(props).get(), "thisisgood");
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

TEST(BasicProperty, setNotValid_03_n) {
  BananaTypes q;
  EXPECT_THROW(q.set({1, 2}), std::invalid_argument);
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

TEST(BasicProperty, fromStringNotValid_04_n) {
  BananaTypes q;
  EXPECT_THROW(nntrainer::from_string("invalid_str", q), std::invalid_argument);
}

TEST(Exporter, invalidMethods_n) {
  auto props = std::make_tuple(NumBanana(), QualityOfBanana());

  nntrainer::Exporter e;
  EXPECT_THROW(e.save_result(props, nntrainer::ExportMethods::METHOD_UNDEFINED),
               nntrainer::exception::not_supported);
}

TEST(Exporter, notExported_n) {
  auto props = std::make_tuple(NumBanana(), QualityOfBanana());

  nntrainer::Exporter e;
  /// intended comment
  // e.save_result(props, nntrainer::ExportMethods::METHOD_STRINGVECTOR);

  EXPECT_THROW(e.get_result<nntrainer::ExportMethods::METHOD_STRINGVECTOR>(),
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
