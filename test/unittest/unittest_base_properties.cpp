// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_base_properties.h
 * @date 09 April 2021
 * @brief This file contains test and specification of properties and exporter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <gtest/gtest.h>

#include <array>
#include <utility>
#include <vector>

#include <base_properties.h>
#include <fc_layer.h>
#include <layer_node.h>
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
  NumBanana(int num = 1) : Property<int>(num) {}   /**< default value if any */
  static constexpr const char *key = "num_banana"; /**< unique key to access */
  using prop_tag = banana_prop_tag;                /**< property type */

  bool isValid(const int &v) const override { return v >= 0; }
};

/**
 * @brief QualityOfBanana property for example, this has to end with "good"
 *
 */
class QualityOfBanana : public nntrainer::Property<std::string> {
public:
  QualityOfBanana(const char *value = "") : Property<std::string>(value) {}
  static constexpr const char *key = "quality_banana";
  using prop_tag = nntrainer::str_prop_tag;

  bool isValid(const std::string &v) const override {
    /// assuming quality of banana property must ends with word "good";
    return nntrainer::endswith(v, "good");
  }
};

/**
 * @brief DimensionOfBanana property for example, this has to have batch size of
 * 1
 *
 */
class DimensionOfBanana : public nntrainer::Property<nntrainer::TensorDim> {
public:
  static constexpr const char *key = "banana_size";
  using prop_tag = nntrainer::dimension_prop_tag;

  bool isValid(const nntrainer::TensorDim &dim) const override {
    std::cerr << dim;
    return dim.batch() == 1;
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
    using T = nntrainer::tag_cast<banana_prop_tag, nntrainer::float_prop_tag,
                                  nntrainer::int_prop_tag>::type;
    ::testing::StaticAssertTypeEq<T, nntrainer::int_prop_tag>();
  }

  { /**< tag_cast does not have appropriate candidates */
    using T = nntrainer::tag_cast<banana_prop_tag, int, std::string>::type;
    ::testing::StaticAssertTypeEq<T, banana_prop_tag>();
  }
}

TEST(BasicProperty, propInfo) {
  { /**< prop_info test */
    using prop_type = nntrainer::prop_info<QualityOfBanana>::prop_type;
    ::testing::StaticAssertTypeEq<prop_type, QualityOfBanana>();

    using tag_type = nntrainer::prop_info<QualityOfBanana>::tag_type;
    ::testing::StaticAssertTypeEq<tag_type, nntrainer::str_prop_tag>();

    using data_type = nntrainer::prop_info<QualityOfBanana>::data_type;
    ::testing::StaticAssertTypeEq<data_type, std::string>();
  }
}

/// @todo convert this to typed param test
TEST(BasicProperty, valid_p) {
  { /** set -> get / to_string, int*/
    NumBanana b;
    b.set(123);
    int ib = b;
    EXPECT_EQ(ib, 123);
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
    std::string sq = q;
    EXPECT_EQ(sq, "this is good");
    EXPECT_EQ(q.get(), "this is good");
    EXPECT_EQ(nntrainer::to_string(q), "this is good");
  }

  { /**< from_string -> get / to_string, string prop */
    QualityOfBanana q;
    nntrainer::from_string("this is good", q);
    EXPECT_EQ(q.get(), "this is good");
    EXPECT_EQ(nntrainer::to_string(q), "this is good");
  }

  { /** set -> get / to_string, dimension*/
    DimensionOfBanana q;
    q.set({1, 2, 3, 4});
    EXPECT_EQ(q.get(), nntrainer::TensorDim(1, 2, 3, 4));
    EXPECT_EQ(nntrainer::to_string(q), "1:2:3:4");
  }

  { /**< from_string -> get / to_string, dimension */
    DimensionOfBanana q;
    nntrainer::from_string("1:2:3:4", q);
    EXPECT_EQ(q.get(), nntrainer::TensorDim(1, 2, 3, 4));
    EXPECT_EQ(nntrainer::to_string(q), "1:2:3:4");
  }

  { /**< from_string -> get / to_string, dimension */
    DimensionOfBanana q;
    nntrainer::from_string("3:4", q);
    EXPECT_EQ(q.get(), nntrainer::TensorDim(1, 1, 3, 4));
    EXPECT_EQ(nntrainer::to_string(q), "1:1:3:4");
  }

  { /**< from_string -> get / to_string, uint vector prop */
    std::vector<NumBanana> bananas;
    EXPECT_EQ(nntrainer::getPropKey(bananas), "num_banana");
    nntrainer::from_string("1, 2,3, 4, 5", bananas);
    auto expected = std::vector<NumBanana>({1, 2, 3, 4, 5});
    EXPECT_EQ(bananas, expected);
    EXPECT_EQ(nntrainer::to_string(bananas), "1,2,3,4,5");
  }

  { /**< from_string -> get / to_string, uint array prop */
    std::array<NumBanana, 4> bananas;
    EXPECT_EQ(nntrainer::getPropKey(bananas), "num_banana");
    nntrainer::from_string("1, 2,3, 4", bananas);
    auto expected = std::array<NumBanana, 4>({1, 2, 3, 4});
    EXPECT_EQ(bananas, expected);
    EXPECT_EQ(nntrainer::to_string(bananas), "1,2,3,4");
  }

  { /**< exporter test */
    auto props = std::make_tuple(NumBanana(), QualityOfBanana());

    nntrainer::Exporter e;
    e.saveResult(props, nntrainer::ExportMethods::METHOD_STRINGVECTOR);

    auto result =
      std::move(e.getResult<nntrainer::ExportMethods::METHOD_STRINGVECTOR>());

    auto pair = std::pair<std::string, std::string>("num_banana", "1");
    EXPECT_EQ(result->at(0), pair);

    auto pair2 = std::pair<std::string, std::string>("quality_banana", "");
    EXPECT_EQ(result->at(1), pair2);
  }

  { /**< export from layer */
    auto lnode =
      nntrainer::LayerNode(std::make_shared<nntrainer::FullyConnectedLayer>(1));
    nntrainer::Exporter e;
    lnode.export_to(e);

    auto result =
      std::move(e.getResult<nntrainer::ExportMethods::METHOD_STRINGVECTOR>());
    auto pair0 = std::pair<std::string, std::string>("name", "");
    EXPECT_EQ(result->at(0), pair0);
    auto pair1 = std::pair<std::string, std::string>("unit", "1");
    EXPECT_EQ(result->at(1), pair1);
  }

  { /**< load from layer */
    auto props =
      std::make_tuple(NumBanana(), QualityOfBanana(), DimensionOfBanana());

    auto v = nntrainer::loadProperties(
      {"num_banana=2", "quality_banana=thisisgood", "num_banana=42",
       "banana_size=2:2:3", "not_used=key"},
      props);

    EXPECT_EQ(v, std::vector<std::string>{"not_used=key"});
    EXPECT_EQ(std::get<0>(props).get(), 42);
    EXPECT_EQ(std::get<1>(props).get(), "thisisgood");
  }

  { /**< load from layer */
    std::tuple<std::array<NumBanana, 4>, std::vector<QualityOfBanana>> props;

    auto v = nntrainer::loadProperties(
      {"num_banana=1,2, 3 ,4", "quality_banana=thisisgood, thisisverygood",
       "not_used=key"},
      props);

    EXPECT_EQ(v, std::vector<std::string>{"not_used=key"});
    auto expected = std::array<NumBanana, 4>({1, 2, 3, 4});
    EXPECT_EQ(std::get<0>(props), expected);
    auto expected2 =
      std::vector<QualityOfBanana>({"thisisgood", "thisisverygood"});
    EXPECT_EQ(std::get<1>(props), expected2);
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
  DimensionOfBanana d;
  EXPECT_THROW(d.set({3, 3, 2, 4}), std::invalid_argument);
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
  DimensionOfBanana d;
  EXPECT_THROW(nntrainer::from_string("1:1:2:3:5", d), std::invalid_argument);
}

TEST(BasicProperty, fromStringNotValid_05_n) {
  DimensionOfBanana d;
  EXPECT_THROW(nntrainer::from_string("2:2:3:5", d), std::invalid_argument);
}

TEST(BasicProperty, fromStringNotValid_06_n) {
  DimensionOfBanana d;
  EXPECT_THROW(nntrainer::from_string("", d), std::invalid_argument);
}

TEST(BasicProperty, fromStringNotValid_07_n) {
  DimensionOfBanana d;
  EXPECT_THROW(nntrainer::from_string(":2:3:5", d), std::invalid_argument);
}

TEST(BasicProperty, fromStringVectorElementContainNotValidString_n) {
  std::vector<NumBanana> bs;
  EXPECT_THROW(nntrainer::from_string("1, 2, 3, not_valid", bs),
               std::invalid_argument);
}

TEST(BasicProperty, fromStringVectorElementContainNotValidProp_n) {
  std::vector<NumBanana> bs;
  EXPECT_THROW(nntrainer::from_string("1, 2, 3, -1", bs),
               std::invalid_argument);
}

TEST(BasicProperty, fromStringLessArrayElementSize_n) {
  std::array<NumBanana, 4> bs;
  EXPECT_THROW(nntrainer::from_string("1, 2, 3", bs), std::invalid_argument);
}

TEST(BasicProperty, fromStringOverArrayElementSize_n) {
  std::array<NumBanana, 4> bs;
  EXPECT_THROW(nntrainer::from_string("1, 2, 3,4,5", bs),
               std::invalid_argument);
}

TEST(BasicProperty, fromStringArrayNotValidString_n) {
  std::array<NumBanana, 4> bs;
  EXPECT_THROW(nntrainer::from_string("1, 2, invalid, 4", bs),
               std::invalid_argument);
}

TEST(BasicProperty, fromStringArrayNotValidProp_n) {
  std::array<NumBanana, 4> bs;
  EXPECT_THROW(nntrainer::from_string("1, 2, -1, 4", bs),
               std::invalid_argument);
}

TEST(Exporter, invalidMethods_n) {
  auto props = std::make_tuple(NumBanana(), QualityOfBanana());

  nntrainer::Exporter e;
  EXPECT_THROW(e.saveResult(props, nntrainer::ExportMethods::METHOD_UNDEFINED),
               nntrainer::exception::not_supported);
}

TEST(Exporter, notExported_n) {
  auto props = std::make_tuple(NumBanana(), QualityOfBanana());

  nntrainer::Exporter e;
  /// intended comment
  // e.saveResult(props, nntrainer::ExportMethods::METHOD_STRINGVECTOR);

  EXPECT_EQ(e.getResult<nntrainer::ExportMethods::METHOD_STRINGVECTOR>(),
            nullptr);
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
