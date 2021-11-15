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

#include <nntrainer_internal.h>
#include <nntrainer_test_util.h>

static std::string mnist_model_path =
  getResPath("mnist.ini", {"test", "test_models", "models"});

/**
 * @brief mlInference test
 *
 */
class mlInference : public testing::Test {
protected:
  /**
   * @brief SetUp the test case
   *
   */
  void SetUp() override { set_feature_state(SUPPORTED); }

  /**
   * @brief TearDown the test case
   *
   */
  void TearDown() override { set_feature_state(NOT_CHECKED_YET); }
};

static int singleshot_case(ml_tensors_info_h in_info,
                           ml_tensors_info_h out_info) {
  ml_single_h single;
  ml_tensors_data_h in_data, out_data;
  ml_tensors_info_h queried_in_info, queried_out_info;
  int status = 0;

  // 1. open singleshot handle
  status = ml_single_open(&single, mnist_model_path.c_str(), in_info, out_info,
                          ML_NNFW_TYPE_NNTR_INF, ML_NNFW_HW_ANY);
  EXPECT_EQ(status, 0);
  if (status != 0) {
    return status;
  }

  status = ml_single_get_input_info(single, &queried_in_info);
  EXPECT_EQ(status, 0);
  if (status != 0) {
    ml_single_close(single);
    return status;
  }

  status = ml_single_get_output_info(single, &queried_out_info);
  EXPECT_EQ(status, 0);
  if (status != 0) {
    ml_tensors_info_destroy(queried_in_info);
    ml_single_close(single);
    return status;
  }

  // 2. allocate data
  status = ml_tensors_data_create(queried_in_info, &in_data);
  EXPECT_EQ(status, 0);
  if (status != 0) {
    ml_tensors_info_destroy(queried_in_info);
    ml_tensors_info_destroy(queried_out_info);
    ml_single_close(single);
    return status;
  }
  status = ml_tensors_data_create(queried_out_info, &out_data);
  EXPECT_EQ(status, 0);
  if (status != 0) {
    ml_tensors_info_destroy(queried_in_info);
    ml_tensors_info_destroy(queried_out_info);
    ml_tensors_data_destroy(in_data);
    ml_single_close(single);
    return status;
  }

  // 3. invoke
  status = ml_single_invoke_fast(single, in_data, out_data);
  EXPECT_EQ(status, 0);
  if (status != 0) {
    ml_tensors_info_destroy(queried_in_info);
    ml_tensors_info_destroy(queried_out_info);
    ml_tensors_data_destroy(in_data);
    ml_tensors_data_destroy(out_data);
    ml_single_close(single);
    return status;
  }

  // 4. release handles
  status = ml_single_close(single);
  EXPECT_EQ(status, 0);

  status = ml_tensors_info_destroy(queried_in_info);
  EXPECT_EQ(status, 0);

  status = ml_tensors_info_destroy(queried_out_info);
  EXPECT_EQ(status, 0);

  status = ml_tensors_data_destroy(in_data);
  EXPECT_EQ(status, 0);

  status = ml_tensors_data_destroy(out_data);
  EXPECT_EQ(status, 0);

  return status;
}

TEST_F(mlInference, singleshot_p) {
  ml_tensors_info_h in_info, out_info; /**< actual true value */

  int status = 0;
  // 1. actual input/output preparation
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

  status = ml_tensors_info_create(&out_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_count(out_info, 1);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_set_tensor_type(out_info, 0, ML_TENSOR_TYPE_FLOAT32);
  EXPECT_EQ(status, 0);
  const unsigned outdim[] = {10, 1, 1, 1};
  status = ml_tensors_info_set_tensor_dimension(out_info, 0, outdim);
  EXPECT_EQ(status, 0);

  status = singleshot_case(in_info, out_info);
  EXPECT_EQ(status, 0) << "case: in_info, out_info given";

  status = singleshot_case(nullptr, out_info);
  EXPECT_EQ(status, 0) << "case: in_info null, out_info given";

  status = singleshot_case(in_info, nullptr);
  EXPECT_EQ(status, 0) << "case: in_info given, out_info null";

  status = singleshot_case(nullptr, nullptr);
  EXPECT_EQ(status, 0) << "case: in_info null, out_info null";

  status = ml_tensors_info_destroy(in_info);
  EXPECT_EQ(status, 0);
  status = ml_tensors_info_destroy(out_info);
  EXPECT_EQ(status, 0);
}

TEST_F(mlInference, singleshotModelDoesNotExist_n) {
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

TEST_F(mlInference, singleshotNoOutData_n) {
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
