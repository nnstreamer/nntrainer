/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file        unittest_nntrainer_databuffer_file.cpp
 * @date        10 April 2020
 * @brief       Unit test databuffer from file.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include "nntrainer_test_util.h"
#include <databuffer_file.h>
#include <fstream>
#include <nntrainer_error.h>

static const std::string getTestResPath(const std::string &file) {
  return getResPath(file, {"test"});
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, setFeatureSize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  nntrainer::TensorDim dim;
  dim.setTensorDim("32:1:1:62720");
  status = data_buffer.setClassNum(10);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile(nntrainer::DATA_TRAIN,
                                   getTestResPath("trainingSet.dat"));
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setFeatureSize(dim);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, setBatchSize_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setBatchSize(32);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, setBatchSize_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setBatchSize(0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer
 */
TEST(nntrainer_DataBuffer, init_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  nntrainer::TensorDim dim;
  dim.setTensorDim("32:1:1:62720");
  status = data_buffer.setBatchSize(32);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setClassNum(10);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile(nntrainer::DATA_TRAIN,
                                   getTestResPath("trainingSet.dat"));
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    data_buffer.setDataFile(nntrainer::DATA_VAL, getTestResPath("valSet.dat"));
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile(nntrainer::DATA_TEST,
                                   getTestResPath("testSet.dat"));
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setFeatureSize(dim);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.init();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer set number of Class
 */
TEST(nntrainer_DataBuffer, setClassNum_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(3);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setClassNum(0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer set number of Class
 */
TEST(nntrainer_DataBuffer, setClassNum_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setClassNum(0);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data Buffer set train Data file
 */
TEST(nntrainer_DataBuffer, setDataFile_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setDataFile(nntrainer::DATA_TRAIN,
                                   getTestResPath("trainingSet.dat"));
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data Buffer set train Data file
 */
TEST(nntrainer_DataBuffer, setDataFile_02_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setDataFile(nntrainer::DATA_TRAIN, "./no_exist.dat");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Data buffer clear all
 */
TEST(nntrainer_DataBuffer, clear_01_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  nntrainer::TensorDim dim;
  dim.setTensorDim("32:1:1:62720");
  status = data_buffer.setBatchSize(32);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setClassNum(10);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile(nntrainer::DATA_TRAIN,
                                   getTestResPath("trainingSet.dat"));
  ASSERT_EQ(status, ML_ERROR_NONE);
  status =
    data_buffer.setDataFile(nntrainer::DATA_VAL, getTestResPath("valSet.dat"));
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setDataFile(nntrainer::DATA_TEST,
                                   getTestResPath("testSet.dat"));
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.setFeatureSize(dim);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.init();
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.run(nntrainer::BufferType::BUF_TRAIN);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.run(nntrainer::BufferType::BUF_TEST);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.run(nntrainer::BufferType::BUF_VAL);
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.clear();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data buffer partial clear
 */
TEST(nntrainer_DataBuffer, clear_02_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setDataFile(nntrainer::DATA_TEST,
                                   getTestResPath("testSet.dat"));
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.clear(nntrainer::BufferType::BUF_TEST);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data buffer all clear after partial clear
 */
TEST(nntrainer_DataBuffer, clear_03_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setDataFile(nntrainer::DATA_TEST,
                                   getTestResPath("testSet.dat"));
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.clear(nntrainer::BufferType::BUF_TEST);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.clear();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data buffer partial clear after all clear
 */
TEST(nntrainer_DataBuffer, clear_04_p) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.setDataFile(nntrainer::DATA_TEST,
                                   getTestResPath("testSet.dat"));
  ASSERT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.clear(nntrainer::BufferType::BUF_TEST);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = data_buffer.clear();
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Data buffer clear BufferType::BUF_UNKNOWN
 */
TEST(nntrainer_DataBuffer, clear_05_n) {
  int status = ML_ERROR_NONE;
  nntrainer::DataBufferFromDataFile data_buffer;
  status = data_buffer.clear(nntrainer::BufferType::BUF_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

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
