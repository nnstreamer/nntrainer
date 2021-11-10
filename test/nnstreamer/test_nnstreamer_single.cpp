// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file test_nnstreamer_single.cpp
 * @date 10 November 2021
 * @brief NNTrainer filter integrated test with nnstreamer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <nnstreamer-single.h>
#include <nnstreamer.h>

#include <nntrainer_test_util.h>

static std::string mnist_model_path =
  getResPath("mnist.ini", {"test", "test_models", "models"});

TEST(mlInference, singleshot_p) {
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_data, out_data;
  int status = 0;

  // 1. input preparation, in this example, we are not filling actual data
  status = ml_tensors_info_create(&in_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_count(in_info, 1);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_tensor_type(in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ(status, 0);

  /// in dim, out dim is in reverse order from nntrainer in nnstreamer
  const unsigned indim[] = {28, 28, 1, 1};
  status = ml_tensors_info_set_tensor_dimension(in_info, 0, indim);
  EXPECT_EQ(status, 0);

  // 2. output preparation
  status = ml_tensors_info_create(&out_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_count(out_info, 1);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_tensor_type(out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ(status, 0);
  const unsigned outdim[] = {10, 1, 1, 1};
  status = ml_tensors_info_set_tensor_dimension(out_info, 0, outdim);
  EXPECT_EQ(status, 0);

  // 3. open singleshot handle
  status = ml_single_open(&single, mnist_model_path.c_str(), in_info, out_info,
                          ML_NNFW_TYPE_NNTR_INF, ML_NNFW_HW_ANY);
  EXPECT_EQ(status, 0);

  // 4. allocate data
  status = ml_tensors_data_create(in_info, &in_data);
  EXPECT_EQ(status, 0);
  status = ml_tensors_data_create(out_info, &out_data);
  EXPECT_EQ(status, 0);

  // 5. invoke
  status = ml_single_invoke_fast(single, in_data, out_data);
  EXPECT_EQ(status, 0);

  // 6. release handles
  status = ml_single_close(single);
  EXPECT_EQ(status, 0);

  status = ml_tensors_data_destroy(in_data);
  EXPECT_EQ(status, 0);
  status = ml_tensors_data_destroy(out_data);
  EXPECT_EQ(status, 0);

  status = ml_tensors_info_destroy(in_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_destroy(out_info);
  EXPECT_EQ(status, 0);
}

TEST(mlInference, singleshotModelDoesNotExist_n) {
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  int status = 0;

  // 1. input preparation, in this example, we are not filling actual data
  status = ml_tensors_info_create(&in_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_count(in_info, 1);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_tensor_type(in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ(status, 0);

  /// in dim, out dim is in reverse order from nntrainer in nnstreamer
  const unsigned indim[] = {28, 28, 1, 1};
  status = ml_tensors_info_set_tensor_dimension(in_info, 0, indim);
  EXPECT_EQ(status, 0);

  // 2. output preparation
  status = ml_tensors_info_create(&out_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_count(out_info, 1);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_tensor_type(out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ(status, 0);
  const unsigned outdim[] = {10, 1, 1, 1};
  status = ml_tensors_info_set_tensor_dimension(out_info, 0, outdim);
  EXPECT_EQ(status, 0);

  // 3. open singleshot handle which does not exist
  status = ml_single_open(&single, "not existing path", in_info, out_info,
                          ML_NNFW_TYPE_NNTR_INF, ML_NNFW_HW_ANY);
  EXPECT_NE(status, 0);

  // 4. release handles
  status = ml_tensors_info_destroy(in_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_destroy(out_info);
  EXPECT_EQ(status, 0);
}

TEST(mlInference, singleshotNoOutData_n) {
  ml_single_h single;
  ml_tensors_info_h in_info, out_info;
  ml_tensors_info_h in_data;
  int status = 0;

  // 1. input preparation, in this example, we are not filling actual data
  status = ml_tensors_info_create(&in_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_count(in_info, 1);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_tensor_type(in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ(status, 0);

  /// in dim, out dim is in reverse order from nntrainer in nnstreamer
  const unsigned indim[] = {28, 28, 1, 1};
  status = ml_tensors_info_set_tensor_dimension(in_info, 0, indim);
  EXPECT_EQ(status, 0);

  // 2. output preparation
  status = ml_tensors_info_create(&out_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_count(out_info, 1);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_tensor_type(out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ(status, 0);
  const unsigned outdim[] = {10, 1, 1, 1};
  status = ml_tensors_info_set_tensor_dimension(out_info, 0, outdim);
  EXPECT_EQ(status, 0);

  // 3. open singleshot handle
  status = ml_single_open(&single, mnist_model_path.c_str(), in_info, out_info,
                          ML_NNFW_TYPE_NNTR_INF, ML_NNFW_HW_ANY);
  EXPECT_EQ(status, 0);

  // 4. allocate data
  status = ml_tensors_data_create(in_info, &in_data);
  EXPECT_EQ(status, 0);

  // 5. invoke
  status = ml_single_invoke_fast(single, in_data, nullptr);
  EXPECT_NE(status, 0);

  // 6. release handles
  status = ml_single_close(single);
  EXPECT_EQ(status, 0);

  status = ml_tensors_data_destroy(in_data);
  EXPECT_EQ(status, 0);

  status = ml_tensors_info_destroy(in_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_destroy(out_info);
  EXPECT_EQ(status, 0);
}
