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
 * @file        unittest_tizen_capi_layer.cc
 * @date        03 April 2020
 * @brief       Unit test utility for layer.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include <gtest/gtest.h>

#include <nntrainer.h>
#include <nntrainer_internal.h>
#include <nntrainer_test_util.h>

/**
 * @brief Neural Network Layer Create / Delete Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, create_delete_01_p) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Create / Delete Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, create_delete_02_p) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Create / Delete Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, create_delete_03_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_UNKNOWN);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_01_p) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property(handle, "input_shape=1:1:6270", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    handle, "normalization=true | standardization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_02_p) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property(handle, "unit=10", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    handle, "bias_initializer=zeros | activation=sigmoid", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_03_p) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property(handle, "activation= sigmoid", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_04_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property(handle, "input_shape=0:0:1", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_05_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property(handle, "epsilon =0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_06_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property(handle, "epsilon =0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_07_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  /** Setting empty property results in error */
  status = ml_train_layer_set_property(handle, "activation=", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_08_p) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_layer_set_property(handle, "weight_regularizer=l2norm",
                                "weight_regularizer_constant=0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negitive test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_09_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status =
    ml_train_layer_set_property(handle, "weight_regularizer=asdfasd",
                                "weight_regularizer_constant=0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_10_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);
  /**
   * Default to none activation if no activation is set.
   * If activation is set which is not available, then error.
   */
  status = ml_train_layer_set_property(handle, "activation=0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_11_n) {
  ml_train_layer_h handle = nullptr;
  int status;
  /**
   * Default to none activation if no activation is set.
   * If activation is set which is not available, then error.
   */
  status = ml_train_layer_set_property(handle, "activation=0.0001", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_12_n) {
  ml_train_layer_h handle = nullptr;
  int status;
  /**
   * If property is set which is an inappropriate way, then error.
   */
  status = ml_train_layer_set_property(handle, "relu", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_13_n) {
  ml_train_layer_h handle = nullptr;
  int status;
  /**
   * If property is set which is an inappropriate way, then error.
   */
  status = ml_train_layer_set_property(handle, "=relu", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_14_n) {
  ml_train_layer_h handle = nullptr;
  int status;
  /**
   * If property is set which is an inappropriate way, then error.
   */
  status = ml_train_layer_set_property(handle, "=0.01", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Set Property Test (negative test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_15_n) {
  ml_train_layer_h handle = nullptr;
  int status;
  /**
   * If property is set which is an inappropriate way, then error.
   */
  status = ml_train_layer_set_property(handle, "activation:relu", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Layer Set Property Test (positive test)
 */
TEST(nntrainer_capi_nnlayer, setproperty_with_single_param_01_p) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property_with_single_param(
    handle, "input_shape=1:1:6270 | normalization=true | standardization=true");
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_with_single_param_02_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property_with_single_param(
    handle, "input_shape=1:1:6270, normalization=true, standardization=true");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_with_single_param_03_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property_with_single_param(
    handle, "input_shape=1:1:6270 ! normalization=true ! standardization=true");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_with_single_param_04_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property_with_single_param(
    handle, "input_shape=1:1:6270 / normalization=true / standardization=true");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_with_single_param_05_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property_with_single_param(
    handle,
    "input_shape=1:1:6270 // normalization=true // standardization=true");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Layer Set Property Test (negative test )
 */
TEST(nntrainer_capi_nnlayer, setproperty_with_single_param_06_n) {
  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_layer_set_property_with_single_param(
    handle, "input_shape=1:1:6270 : normalization=true : standardization=true");
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/*** since tizen 6.5 ***/

/**
 * @brief CapiLayerPropertyTester
 * @tparam ml_train_layer_type_e layer type to create
 * @tparam const char * valid property
 * @tparam bool if setting property should success or fail
 */
class nntrainerCapiLayerTester
  : public ::testing::TestWithParam<std::tuple<
      ml_train_layer_type_e, const std::vector<const char *>, bool>> {};

/**
 * @brief layer creating and setting property, this is eithr positive or
 * negative
 */
TEST_P(nntrainerCapiLayerTester, layer_create_and_set_property) {
  auto param = GetParam();
  auto type = std::get<0>(param);
  auto props = std::get<1>(param);
  auto should_success = std::get<2>(param);

  ml_train_layer_h handle;
  int status;
  status = ml_train_layer_create(&handle, type);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = 0;
  for (auto &i : props) {
    int ret = ml_train_layer_set_property(handle, i, NULL);
    if (ret != ML_ERROR_NONE) {
      std::cerr << "setting property failed at: " << i << '\n';
    }
    status += ret;
  }
  EXPECT_TRUE((status == ML_ERROR_NONE) == should_success);

  status = ml_train_layer_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

static std::tuple<ml_train_layer_type_e, const std::vector<const char *>, bool>
mkTc(ml_train_layer_type_e type, const std::vector<const char *> &vec,
     bool success) {
  return std::make_tuple(type, vec, success);
}

GTEST_PARAMETER_TEST(
  CapiPropertyTest, nntrainerCapiLayerTester,
  ::testing::Values(
    mkTc(ML_TRAIN_LAYER_TYPE_BN,
         {"name=bn", "epsilon=0.001", "moving_mean_initializer=zeros",
          "moving_variance_initializer=ones", "gamma_initializer=zeros",
          "beta_initializer=ones", "momentum=0.9"},
         true),
    mkTc(ML_TRAIN_LAYER_TYPE_BN, {"name=bn", "epsilon=no_float"}, false),
    mkTc(ML_TRAIN_LAYER_TYPE_CONV2D,
         {"filters=3", "kernel_size=2, 2", "stride=1,1", "padding=0, 0"}, true),
    mkTc(ML_TRAIN_LAYER_TYPE_CONV2D, {"kernel_size=string"}, false),
    mkTc(ML_TRAIN_LAYER_TYPE_POOLING2D, {"pooling=max", "pool_size=1, 1"},
         true),
    mkTc(ML_TRAIN_LAYER_TYPE_POOLING2D, {"pooling=undef", "pool_size=1, 1"},
         false),
    mkTc(ML_TRAIN_LAYER_TYPE_FLATTEN, {"name=flat"}, true),
    mkTc(ML_TRAIN_LAYER_TYPE_ACTIVATION, {"activation=relu"}, true),
    mkTc(ML_TRAIN_LAYER_TYPE_ACTIVATION, {"activation=undef"}, false)));

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

  /** ignore tizen feature check while running the testcases */
  set_feature_state(ML_FEATURE, SUPPORTED);
  set_feature_state(ML_FEATURE_INFERENCE, SUPPORTED);
  set_feature_state(ML_FEATURE_TRAINING, SUPPORTED);

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }

  /** reset tizen feature check state */
  set_feature_state(ML_FEATURE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_INFERENCE, NOT_CHECKED_YET);
  set_feature_state(ML_FEATURE_TRAINING, NOT_CHECKED_YET);

  return result;
}
