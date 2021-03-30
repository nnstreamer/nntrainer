// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file	 unittest_app_context.h
 * @date	 9 November 2020
 * @brief	 This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see		 https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <gtest/gtest.h>

#include <fstream>
#include <memory>
#include <typeinfo>
#include <unistd.h>

#include <optimizer_devel.h>
#include <weight.h>

#include <app_context.h>
#include <nntrainer_error.h>

class nntrainerAppContextDirectory : public ::testing::Test {

protected:
  void SetUp() override {
    int status = mkdir("testdir", 0777);
    ASSERT_EQ(status, 0);

    std::ofstream file("testdir/testfile.txt");
    ASSERT_EQ(file.fail(), false);

    file << "testdata";
    ASSERT_EQ(file.fail(), false);

    file.close();

    char buf[2048];
    char *ret = getcwd(buf, 2048);
    ASSERT_NE(ret, nullptr);
    current_directory = std::string(buf);
  }

  void TearDown() override {
    int status = remove("testdir/testfile.txt");
    ASSERT_EQ(status, 0);

    status = rmdir("testdir");
    ASSERT_EQ(status, 0);
  }

  std::string current_directory;
};

TEST_F(nntrainerAppContextDirectory, readFromGetPath_p) {
  nntrainer::AppContext ac = nntrainer::AppContext::Global();

  std::string path = ac.getWorkingPath("testfile.txt");
  EXPECT_EQ(path, "testfile.txt");

  ac.setWorkingDirectory("testdir");

  path = ac.getWorkingPath("testfile.txt");
  EXPECT_EQ(path, current_directory + "/testdir/testfile.txt");

  std::ifstream file(path);
  std::string s;
  file >> s;
  EXPECT_EQ(s, "testdata");

  file.close();

  path = ac.getWorkingPath("/absolute/path");
  EXPECT_EQ(path, "/absolute/path");

  path = ac.getWorkingPath("");
  EXPECT_EQ(path, current_directory + "/testdir");
}

TEST_F(nntrainerAppContextDirectory, notExisitingSetDirectory_n) {
  nntrainer::AppContext ac = nntrainer::AppContext::Global();

  EXPECT_THROW(ac.setWorkingDirectory("testdir_does_not_exist"),
               std::invalid_argument);
}

class CustomOptimizer : public nntrainer::Optimizer {
public:
  /** Full custom optimizer example which overrides all functions */
  const std::string getType() const override { return "identity_optimizer"; }

  float getLearningRate() const override { return 1.0f; }

  double getLearningRate(size_t iteration) const override { return 1.0f; }

  int setProperty(std::vector<std::string> values) override { return 1; }

  int initialize() override { return 0; }

  void addOptimizerVariable(std::vector<nntrainer::Weight> &params) override {}

  void setProperty(const PropertyType type, const std::string &value = "") override {}

  void checkValidation() const override {}

  void applyGradient(nntrainer::Weight &weight, double updated_lr,
                     int iteration) override {}
};

class CustomOptimizer2 : public nntrainer::Optimizer {
public:
  /** Minimal custom optimizer example which define only necessary functions */
  const std::string getType() const override { return "identity_optimizer"; }

  int initialize() override { return 0; }

  double getLearningRate(size_t iteration) const override { return 1.0f; }

  void addOptimizerVariable(std::vector<nntrainer::Weight> &params) override {}

  void applyGradient(nntrainer::Weight &weight, double updated_lr,
                     int iteration) override {}
};

/// @todo solidify the api signature
class CustomLayer : public ml::train::Layer {
public:
  static const std::string type;

  int setProperty(std::vector<std::string> values) override { return 1; }

  void setProperty(const PropertyType type, const std::string &value = "") override {}

  int checkValidation() override { return 1; }

  float getLoss() override { return 0.0f; }

  void setTrainable(bool train) override {}

  bool getFlatten() override { return true; }

  std::string getName() noexcept override { return ""; }

  const std::string getType() const override { return CustomLayer::type; }

  void printPreset(std::ostream &out, PrintPreset preset) override {}
};

const std::string CustomLayer::type = "identity_layer";

using AC = nntrainer::AppContext;

AC::PtrType<ml::train::Optimizer>
createCustomOptimizer(const AC::PropsType &v) {
  auto p = std::make_unique<CustomOptimizer>();
  p->setProperty(v);
  return p;
}

/// @todo change this to TEST_P to add other types of object
TEST(nntrainerAppContextObjs, RegisterCreateCustomOptimizer_p) {

  // register without key in this case, getType() will be called and used
  {
    auto ac = nntrainer::AppContext();
    int num_id = ac.registerFactory(createCustomOptimizer);
    auto opt = ac.createObject<ml::train::Optimizer>("identity_optimizer", {});
    EXPECT_EQ(typeid(*opt).hash_code(), typeid(CustomOptimizer).hash_code());
    opt = ac.createObject<ml::train::Optimizer>(num_id, {});
    EXPECT_EQ(typeid(*opt).hash_code(), typeid(CustomOptimizer).hash_code());
  }

  // register with key
  {
    auto ac = nntrainer::AppContext();
    int num_id = ac.registerFactory(createCustomOptimizer, "custom_key");
    auto opt = ac.createObject<ml::train::Optimizer>("custom_key", {});
    EXPECT_EQ(typeid(*opt).hash_code(), typeid(CustomOptimizer).hash_code());
    opt = ac.createObject<ml::train::Optimizer>(num_id, {});
    EXPECT_EQ(typeid(*opt).hash_code(), typeid(CustomOptimizer).hash_code());
  }

  // register with key and custom id
  {
    auto ac = nntrainer::AppContext();
    int num_id = ac.registerFactory(createCustomOptimizer, "custom_key", 5);
    EXPECT_EQ(num_id, 5);
    auto opt = ac.createObject<ml::train::Optimizer>("custom_key", {});
    EXPECT_EQ(typeid(*opt).hash_code(), typeid(CustomOptimizer).hash_code());
    opt = ac.createObject<ml::train::Optimizer>(num_id, {});
    EXPECT_EQ(typeid(*opt).hash_code(), typeid(CustomOptimizer).hash_code());
  }
}

TEST(nntrainerAppContextObjs, RegisterFactoryWithClashingKey_n) {
  auto ac = nntrainer::AppContext();

  ac.registerFactory(createCustomOptimizer, "custom_key");

  EXPECT_THROW(ac.registerFactory(createCustomOptimizer, "custom_key"),
               std::invalid_argument);
}

TEST(nntrainerAppContextObjs, RegisterFactoryWithClashingIntKey_n) {
  auto ac = nntrainer::AppContext();

  ac.registerFactory(createCustomOptimizer, "custom_key", 3);
  EXPECT_THROW(ac.registerFactory(createCustomOptimizer, "custom_other_key", 3),
               std::invalid_argument);
}

TEST(nntrainerAppContextObjs, RegisterFactoryWithClashingAutoKey_n) {
  auto ac = nntrainer::AppContext();

  ac.registerFactory(createCustomOptimizer);
  EXPECT_THROW(ac.registerFactory(createCustomOptimizer),
               std::invalid_argument);
}

TEST(nntrainerAppContextObjs, createObjectNotExistingKey_n) {
  auto ac = nntrainer::AppContext();

  ac.registerFactory(createCustomOptimizer);
  EXPECT_THROW(ac.createObject<ml::train::Optimizer>("not_exisiting_key"),
               nntrainer::exception::not_supported);
}

TEST(nntrainerAppContextObjs, createObjectNotExistingIntKey_n) {
  auto ac = nntrainer::AppContext();

  int num = ac.registerFactory(createCustomOptimizer);
  EXPECT_THROW(ac.createObject<ml::train::Optimizer>(num + 3),
               nntrainer::exception::not_supported);
}

TEST(nntrainerAppContextObjs, callingUnknownFactoryOptimizerWithKey_n) {
  auto ac = nntrainer::AppContext();

  int num = ac.registerFactory(
    nntrainer::AppContext::unknownFactory<ml::train::Optimizer>, "unknown",
    999);

  EXPECT_EQ(num, 999);
  EXPECT_THROW(ac.createObject<ml::train::Optimizer>("unknown"),
               std::runtime_error);
}

TEST(nntrainerAppContextObjs, callingUnknownFactoryOptimizerWithIntKey_n) {
  auto ac = nntrainer::AppContext();

  int num = ac.registerFactory(
    nntrainer::AppContext::unknownFactory<ml::train::Optimizer>, "unknown",
    999);

  EXPECT_EQ(num, 999);
  EXPECT_THROW(ac.createObject<ml::train::Optimizer>(num), std::runtime_error);
}

/**
 * @brief Main gtest
 */
int main(int argc, char **argv) {
  int result = -1;

  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }

  return result;
}
