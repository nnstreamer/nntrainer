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
#include <unistd.h>

#include <app_context.h>

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
