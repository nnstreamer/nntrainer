// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file properties.h
 * @date 03 Aug 2020
 * @brief Properties handler for NNtrainer.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <string>

#include <gtest/gtest.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <properties.h>
#include <vector>

namespace Example {
enum class FruitType {
  POME,
  DRUPE,
  BERRY,
  MELON,
  CITRUS,
  TROPICAL,
  UNKNOWN,
};

enum class FruitColor {
  WHITE,
  ORANGE,
  RED,
  GREEN,
  BLACK,
  UNKNOWN,
};

struct FruitSize {};

} // namespace Example

namespace nntrainer {
/**
 * @brief trait specialization part
 */
template <> struct prop_traits<Example::FruitType> {

  static constexpr bool is_property = true;

  typedef Example::FruitType value_type;

  static std::string getKey() { return "type"; }

  static bool is_valid(value_type c) { return c != value_type::UNKNOWN; }

  static value_type getDefault() { return value_type::UNKNOWN; }

  static std::string serialize(value_type c) {
    switch (c) {
    case value_type::POME:
      return "pome";
    case value_type::DRUPE:
      return "drupe";
    case value_type::BERRY:
      return "berry";
    case value_type::MELON:
      return "melon";
    case value_type::CITRUS:
      return "citrus";
    case value_type::TROPICAL:
      return "tropical";
    case value_type::UNKNOWN:
      return "unknown";
    default:
      throw std::invalid_argument("given parameter is not serializable");
    }
  }

  static value_type deserialize(const std::string &val) {
    if (val == "pome")
      return value_type::POME;
    if (val == "drupe")
      return value_type::DRUPE;
    if (val == "berry")
      return value_type::BERRY;
    if (val == "melon")
      return value_type::MELON;
    if (val == "citrus")
      return value_type::CITRUS;
    if (val == "tropical")
      return value_type::TROPICAL;
    if (val == "unknown")
      return value_type::UNKNOWN;
    throw std::invalid_argument("given value is not deserializable");
  }
};

template <> struct prop_traits<Example::FruitColor> {
  static constexpr bool is_property = true;

  typedef Example::FruitColor value_type;

  static std::string getKey() { return "color"; }

  static bool is_valid(value_type c) { return c != value_type::UNKNOWN; }

  static value_type getDefault() { return value_type::UNKNOWN; }

  static std::string serialize(value_type c) {
    switch (c) {
    case value_type::WHITE:
      return "white";
    case value_type::ORANGE:
      return "orange";
    case value_type::RED:
      return "red";
    case value_type::GREEN:
      return "green";
    case value_type::BLACK:
      return "black";
    case value_type::UNKNOWN:
      return "unknown";
    default:
      /// should not reach here
      throw std::invalid_argument("c is not serializable");
    }
  }

  static value_type deserialize(const std::string &val) {
    if (val == "white")
      return value_type::WHITE;
    if (val == "orange")
      return value_type::ORANGE;
    if (val == "red")
      return value_type::RED;
    if (val == "green")
      return value_type::GREEN;
    if (val == "black")
      return value_type::BLACK;
    if (val == "unknown")
      return value_type::UNKNOWN;
    throw std::invalid_argument("given value is not deserializable");
  }
};

template <> struct prop_traits<Example::FruitSize> {
  static constexpr bool is_property = true;

  typedef int value_type;

  static std::string getKey() { return "size"; }

  static bool is_valid(value_type c) { return c > 0; }

  static value_type getDefault() { return 0; }

  static std::string serialize(value_type c) { return std::to_string(c); }

  static value_type deserialize(const std::string &val) {
    return std::stoi(val.c_str());
  }
};
} // namespace nntrainer

namespace prop_test {
template <typename T> struct test_cases {
  static const std::vector<T> validValues() {
    throw std::runtime_error("no test case available");
  } /**< valid values to be tested */

  static const std::vector<std::string> validStrings() {
    throw std::runtime_error("no test case available");
  } /**< valid strings to be tested, should be aligned with valid_values */

  static const std::vector<T> notValidValues() {
    throw std::runtime_error("no test case available");
  } /**< invalid values for negative cases (should throw error when check valid)
     */

  static const std::vector<std::string> notValidStrings() {
    throw std::runtime_error("no test case available");
  } /**< invalid strings for negative cases (should NOT throw error when
       deserialize, should throw error when check valid) */

  static const std::vector<std::string> erronuousStrings() {
    throw std::runtime_error("no test case available");
  } /**< invalid strings for negative cases (should throw error because it is
       not deserializable) */
};

template <> struct test_cases<Example::FruitType> {
  typedef nntrainer::prop_traits<Example::FruitType>::value_type value_type;

  static const std::vector<value_type> validValues() {
    return {value_type::POME,  value_type::DRUPE,  value_type::BERRY,
            value_type::MELON, value_type::CITRUS, value_type::TROPICAL};
  }

  static const std::vector<std::string> validStrings() {
    return {"pome", "drupe", "berry", "melon", "citrus", "tropical"};
  }

  static const std::vector<value_type> notValidValues() {
    return {value_type::UNKNOWN};
  }

  static const std::vector<std::string> notValidStrings() {
    return {"unknown"};
  }

  static const std::vector<std::string> erronuousStrings() {
    return {"pame", "drape", "verry"};
  }
};

template <> struct test_cases<Example::FruitColor> {
  typedef nntrainer::prop_traits<Example::FruitColor>::value_type value_type;

  static const std::vector<value_type> validValues() {
    return {value_type::WHITE, value_type::ORANGE, value_type::RED,
            value_type::GREEN, value_type::BLACK};
  }

  static const std::vector<std::string> validStrings() {
    return {"white", "orange", "red", "green", "black"};
  }

  static const std::vector<value_type> notValidValues() {
    return {value_type::UNKNOWN};
  }

  static const std::vector<std::string> notValidStrings() {
    return {"unknown"};
  }

  static const std::vector<std::string> erronuousStrings() {
    return {"whate", "abcd", "kdb"};
  }
};

template <> struct test_cases<Example::FruitSize> {
  typedef nntrainer::prop_traits<Example::FruitSize>::value_type value_type;

  static const std::vector<value_type> validValues() {
    return {1, 2, 3, 4, 5, 6};
  }

  static const std::vector<std::string> validStrings() {
    return {"1", "2", "3", "4", "5", "6"};
  }

  static const std::vector<value_type> notValidValues() { return {0, -1, -2}; }

  static const std::vector<std::string> notValidStrings() {
    return {"0", "-1", "-2"};
  }

  static const std::vector<std::string> erronuousStrings() {
    return {"i_am", "not ", "integer "};
  }
};
} // namespace prop_test

template <typename T> class PropertyTest : public testing::Test {};

/** todo add test case for unregistered proptraits */

TYPED_TEST_CASE_P(PropertyTest);

TYPED_TEST_P(PropertyTest, validValuesAreValid_p) {
  for (auto &i : prop_test::test_cases<TypeParam>::validValues())
    EXPECT_TRUE(nntrainer::prop_traits<TypeParam>::is_valid(i));
}

TYPED_TEST_P(PropertyTest, validValuesAreSerializable_p) {
  auto v = prop_test::test_cases<TypeParam>::validValues();
  auto s = prop_test::test_cases<TypeParam>::validStrings();

  for (size_t i = 0; i < v.size(); i++)
    EXPECT_EQ(nntrainer::prop_traits<TypeParam>::serialize(v[i]), s[i]);
}

TYPED_TEST_P(PropertyTest, validStringsAreDeserializable_p) {
  auto v = prop_test::test_cases<TypeParam>::validValues();
  auto s = prop_test::test_cases<TypeParam>::validStrings();

  for (size_t i = 0; i < v.size(); i++)
    EXPECT_EQ(v[i], nntrainer::prop_traits<TypeParam>::deserialize(s[i]));
}

TYPED_TEST_P(PropertyTest, notValidValuesAreNotValid_n) {
  for (auto &i : prop_test::test_cases<TypeParam>::notValidValues())
    EXPECT_FALSE(nntrainer::prop_traits<TypeParam>::is_valid(i));
}

TYPED_TEST_P(PropertyTest, notValidValuesAreSerializable_n) {
  auto v = prop_test::test_cases<TypeParam>::notValidValues();
  auto s = prop_test::test_cases<TypeParam>::notValidStrings();

  for (size_t i = 0; i < v.size(); i++)
    EXPECT_EQ(nntrainer::prop_traits<TypeParam>::serialize(v[i]), s[i]);
}

TYPED_TEST_P(PropertyTest, notValidStringsAreDeserializable_n) {
  auto v = prop_test::test_cases<TypeParam>::notValidValues();
  auto s = prop_test::test_cases<TypeParam>::notValidStrings();

  for (size_t i = 0; i < v.size(); i++)
    EXPECT_EQ(v[i], nntrainer::prop_traits<TypeParam>::deserialize(s[i]));
}

TYPED_TEST_P(PropertyTest, errorneuousStringThrowInvalidArgument_n) {
  for (auto &i : prop_test::test_cases<TypeParam>::erronuousStrings())
    EXPECT_THROW(nntrainer::prop_traits<TypeParam>::deserialize(i),
                 std::invalid_argument);
}

TYPED_TEST_P(PropertyTest, singlePropertiesSetGetValidType_p) {
  nntrainer::Properties<TypeParam> props;
  for (auto &i : prop_test::test_cases<TypeParam>::validValues()) {
    EXPECT_NO_THROW(props.template set<TypeParam>(i));
    EXPECT_EQ(props.template get<TypeParam>(), i);
  }
}

TYPED_TEST_P(PropertyTest, singlePropertiesSetGetValidString_p) {
  nntrainer::Properties<TypeParam> props;
  std::string key = nntrainer::prop_traits<TypeParam>::getKey();

  for (auto &i : prop_test::test_cases<TypeParam>::validStrings()) {
    EXPECT_NO_THROW(props.set(key, i));
    EXPECT_EQ(props.get(key), i);
  }
}

// TYPED_TEST_P(PropertyTest,
// singlePropertiesSetErroneuousStringThrowInvalidArgument_n){

// }

// TYPED_TEST_P(PropertyTest,
// singlePropertiesSetInvalidStringThrowInvalidArgument_n){

// }

// TYPED_TEST_P(PropertyTest, singlePropertiesGetInValidType_n) {
//   /// This will make a compile time error so commented
// }

// TYPED_TEST_P(PropertyTest, singlePropertiesGetValidInvalidKey_n) {

// }

REGISTER_TYPED_TEST_CASE_P(
  PropertyTest, validValuesAreValid_p, validValuesAreSerializable_p,
  validStringsAreDeserializable_p, notValidValuesAreNotValid_n,
  notValidValuesAreSerializable_n, notValidStringsAreDeserializable_n,
  errorneuousStringThrowInvalidArgument_n, singlePropertiesSetGetValidType_p,
  singlePropertiesSetGetValidString_p);

using ExamplePropertyTypes =
  ::testing::Types<Example::FruitType, Example::FruitColor, Example::FruitSize>;

INSTANTIATE_TYPED_TEST_CASE_P(ExampleTest, PropertyTest, ExamplePropertyTypes);

/// @todo instantiate test case with real types (from nntrainer)

/// @todo Add parametrized test for proptypes

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    ml_loge("Failed to init gtest\n");
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    ml_loge("Failed to run test.\n");
  }

  return result;
}
