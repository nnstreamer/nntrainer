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
  QualityOfBanana() : nntrainer::Property<std::string>() {}
  QualityOfBanana(const char *value) { set(value); }
  static constexpr const char *key = "quality_banana";
  using prop_tag = nntrainer::str_prop_tag;

  bool isValid(const std::string &v) const override {
    /// assuming quality of banana property must ends with word "good";
    return nntrainer::endswith(v, "good");
  }
};

/**
 * @brief Property example to be used as a bool
 *
 */
class MarkAsGoodBanana : public nntrainer::Property<bool> {
public:
  MarkAsGoodBanana(bool val = true) { set(val); } /**< default value if any */
  static constexpr const char *key = "mark_good"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;      /**< property type */
};

/**
 * @brief Property example to be used as a float
 *
 */
class FreshnessOfBanana : public nntrainer::Property<float> {
public:
  FreshnessOfBanana(float val = 0.0) { set(val); } /**< default value if any */
  static constexpr const char *key = "how_fresh";  /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag;      /**< property type */
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
    return dim.batch() == 1;
  }
};

/**
 * @brief Pointer of banana property
 *
 */
class PtrOfBanana : public nntrainer::Property<int *> {
public:
  static constexpr const char *key = "ptr_banana";
  using prop_tag = nntrainer::ptr_prop_tag;
};

/**
 * @brief Enuminformation of BananaType;
 *
 */
struct BananaEnumInfo {
  /**
   * @brief underlying enum
   *
   */
  enum class Enum {
    Cavendish = 0,
    Plantain = 1,
    Manzano = 2,
  };

  static constexpr std::initializer_list<Enum> EnumList = {
    Enum::Cavendish, Enum::Plantain, Enum::Manzano};

  static constexpr const char *EnumStr[] = {"Cavendish", "Plantain", "Manzano"};
};

/**
 * @brief Type of Banana (enum based)
 *
 */
class BananaType : public nntrainer::EnumProperty<BananaEnumInfo> {
public:
  using prop_tag = nntrainer::enum_class_prop_tag;
  static constexpr const char *key = "banana_type";
};

} // namespace

/// @todo convert this to typed param test
TEST(BasicProperty, valid_p) {
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

  { /**< prop_info test */
    using prop_type = nntrainer::prop_info<QualityOfBanana>::prop_type;
    ::testing::StaticAssertTypeEq<prop_type, QualityOfBanana>();

    using tag_type = nntrainer::prop_info<QualityOfBanana>::tag_type;
    ::testing::StaticAssertTypeEq<tag_type, nntrainer::str_prop_tag>();

    using data_type = nntrainer::prop_info<QualityOfBanana>::data_type;
    ::testing::StaticAssertTypeEq<data_type, std::string>();
  }

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

  { /**< from_string -> get / to_string, boolean */
    MarkAsGoodBanana q;
    nntrainer::from_string("true", q);
    EXPECT_EQ(q.get(), true);
    EXPECT_EQ(nntrainer::to_string(q), "true");
  }

  { /** set -> get / to_string, boolean*/
    MarkAsGoodBanana q;
    q.set(true);
    EXPECT_EQ(q.get(), true);
    EXPECT_EQ(nntrainer::to_string(q), "true");
  }

  { /**< from_string -> get / to_string, ptr */
    PtrOfBanana pb;
    int a = 1;
    std::ostringstream ss;
    ss << &a;
    nntrainer::from_string(ss.str(), pb);
    EXPECT_EQ(pb.get(), &a);
    EXPECT_EQ(*pb.get(), a);
    EXPECT_EQ(nntrainer::to_string(pb), ss.str());
  }

  { /** set -> get / to_string, boolean*/
    PtrOfBanana pb;
    int a = 1;
    pb.set(&a);
    EXPECT_EQ(pb.get(), &a);
    EXPECT_EQ(*pb.get(), 1);
    std::ostringstream ss;
    ss << &a;
    EXPECT_EQ(nntrainer::to_string(pb), ss.str());
  }

  { /**< from_string -> get / to_string, float */
    FreshnessOfBanana q;
    nntrainer::from_string("1.3245", q);
    EXPECT_FLOAT_EQ(q.get(), 1.3245f);
  }

  { /** set -> get / to_string, float*/
    FreshnessOfBanana q;
    q.set(1.3245f);
    EXPECT_FLOAT_EQ(q.get(), 1.3245f);
  }

  { /**< enum type test from_string -> get */
    BananaType t;
    nntrainer::from_string("CAVENDISH", t);
    EXPECT_EQ(t.get(), BananaEnumInfo::Enum::Cavendish);
    nntrainer::from_string("Plantain", t);
    EXPECT_EQ(t.get(), BananaEnumInfo::Enum::Plantain);
    nntrainer::from_string("manzano", t);
    EXPECT_EQ(t.get(), BananaEnumInfo::Enum::Manzano);
  }

  { /**< enum type test set -> to_string */
    BananaType t;
    t.set(BananaEnumInfo::Enum::Cavendish);
    EXPECT_EQ("Cavendish", nntrainer::to_string(t));
    t.set(BananaEnumInfo::Enum::Plantain);
    EXPECT_EQ("Plantain", nntrainer::to_string(t));
    t.set(BananaEnumInfo::Enum::Manzano);
    EXPECT_EQ("Manzano", nntrainer::to_string(t));
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
    e.saveResult(props, ml::train::ExportMethods::METHOD_STRINGVECTOR);

    auto result = e.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>();

    auto pair = std::pair<std::string, std::string>("num_banana", "1");
    EXPECT_EQ(result->at(0), pair);
  }

  { /**< export from layer */
    auto lnode = nntrainer::LayerNode(
      std::move(std::make_unique<nntrainer::FullyConnectedLayer>()));
    nntrainer::Exporter e;
    lnode.setProperty({"unit=1"});
    lnode.exportTo(e, ml::train::ExportMethods::METHOD_STRINGVECTOR);

    auto result = e.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>();
    auto pair1 = std::pair<std::string, std::string>("unit", "1");
    for (unsigned int i = 0; i < (*result).size(); ++i) {
      if (result->at(i).first == "unit") {
        EXPECT_EQ(result->at(i), pair1);
      }
    }
  }

  { /**< load from layer */
    auto props =
      std::make_tuple(NumBanana(), QualityOfBanana(), DimensionOfBanana());

    auto v = nntrainer::loadProperties(
      {"num_banana=2 | quality_banana=thisisgood | num_banana=42",
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

TEST(BasicProperty, setNotValid_04_n) {
  DimensionOfBanana d;
  EXPECT_THROW(d.set({1, 2, 3, 4, 5}), std::invalid_argument);
}

TEST(BasicProperty, setNotValid_05_n) {
  DimensionOfBanana d;
  EXPECT_THROW(d.set({0}), std::invalid_argument);
}

TEST(BasicProperty, setNotValid_06_n) {
  DimensionOfBanana d;
  EXPECT_THROW(d.set({0, 1}), std::invalid_argument);
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

TEST(BasicProperty, fromStringNotValid_08_n) {
  MarkAsGoodBanana d;
  EXPECT_THROW(nntrainer::from_string("no", d), std::invalid_argument);
}

TEST(BasicProperty, fromStringNotValid_09_n) {
  FreshnessOfBanana d;
  EXPECT_THROW(nntrainer::from_string("not_float", d), std::invalid_argument);
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
  EXPECT_THROW(e.saveResult(props, ml::train::ExportMethods::METHOD_UNDEFINED),
               nntrainer::exception::not_supported);
}

TEST(Exporter, notExported_n) {
  auto props = std::make_tuple(NumBanana(), QualityOfBanana());

  nntrainer::Exporter e;
  /// intended comment
  // e.saveResult(props, ml::train::ExportMethods::METHOD_STRINGVECTOR);

  EXPECT_EQ(e.getResult<ml::train::ExportMethods::METHOD_STRINGVECTOR>(),
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
