/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
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
 * @file        unittest_tizen_capi.cc
 * @date        03 April 2020
 * @brief       Unit test utility.
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs
 */
#include "nntrainer_test_util.h"
#include <nntrainer.h>

/**
 * @brief Neural Network Model Contruct / Destruct Test (possitive test )
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_01_p) {
  ml_nnmodel_h handle;
  int status;
  status = ml_nnmodel_construct(&handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Destruct Test (negative test )
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_02_n) {
  ml_nnmodel_h handle = NULL;
  int status;
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Construct wit Configuration File Test
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_03_n) {
  ml_nnmodel_h handle;
  const char *model_conf = "/test/cannot_find.ini";
  int status;
  status = ml_nnmodel_construct_with_conf(model_conf, &handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Construct wit Configuration File Test
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_04_p) {
  ml_nnmodel_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_construct_destruct_04_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file);
  status = ml_nnmodel_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_01_p) {
  ml_nnmodel_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file);
  status = ml_nnmodel_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_compile_with_conf(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_02_n) {
  ml_nnmodel_h handle = NULL;
  int status = ML_ERROR_NONE;
  status = ml_nnmodel_construct(&handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_compile_with_conf(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_03_n) {
  ml_nnmodel_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_03_n.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("HiddenSize = 62720", "HiddenSize=0", config_file);
  status = ml_nnmodel_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_compile_with_conf(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, train_01_p) {
  ml_nnmodel_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_train_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("HiddenSize = 62720", "HiddenSize=62720", config_file);
  replaceString("minibatch = 32", "minibatch = 16", config_file);
  replaceString("BufferSize=100", "", config_file);
  status = ml_nnmodel_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_compile_with_conf(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_train_with_file(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_destruct(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_01_p) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layer;

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layer, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layer, "input_shape= 32:1:1:6270", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layer, "normalization = true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_delete(layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_02_p) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layer;

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layer, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layer, "input_shape= 32:1:1:6270",
                                   "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_delete(layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_03_n) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layer;

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layer, ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layer, "input_shape= 32:1:1:62720",
                                   "activation=sigmoid", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_nnlayer_delete(layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_04_p) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layers[2];

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[0], ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_nnlayer_set_property(layers[0], "input_shape= 32:1:1:62720",
                            "normalization=true", "bias_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[1], ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layers[1], "unit= 10", "activation=softmax",
                                   "bias_zero=true", "weight_decay=l2norm",
                                   "weight_decay_lambda=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_delete(layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, create_optimizer_05_p) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layers[2];
  ml_nnopt_h optimizer;

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[0], ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_nnlayer_set_property(layers[0], "input_shape= 32:1:1:62720",
                            "normalization=true", "bias_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[1], ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layers[1], "unit= 10", "activation=softmax",
                                   "bias_zero=true", "weight_decay=l2norm",
                                   "weight_decay_lambda=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_create(&optimizer, "adam");
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_delete(layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnoptimizer_delete(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, compile_04_p) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layers[2];
  ml_nnopt_h optimizer;

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[0], ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_nnlayer_set_property(layers[0], "input_shape= 32:1:1:62720",
                            "normalization=true", "bias_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[1], ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layers[1], "unit= 10", "activation=softmax",
                                   "bias_zero=true", "weight_decay=l2norm",
                                   "weight_decay_lambda=0.005",
                                   "weight_ini=xavier_uniform", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_create(&optimizer, "adam");
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_compile(model, optimizer, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_delete(layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnoptimizer_delete(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, train_with_file_01_p) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layers[2];
  ml_nnopt_h optimizer;

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[0], ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_nnlayer_set_property(layers[0], "input_shape= 16:1:1:62720",
                            "normalization=true", "bias_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[1], ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layers[1], "unit= 10", "activation=softmax",
                                   "bias_zero=true", "weight_decay=l2norm",
                                   "weight_decay_lambda=0.005",
                                   "weight_ini=xavier_uniform", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_create(&optimizer, "adam");
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_compile(model, optimizer, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_train_with_file(
    model, "epochs=1", "batch_size=16", "train_data=trainingSet.dat",
    "val_data=valSet.dat", "label_data=label.dat", "buffer_size=100",
    "model_file=model.bin", NULL);

  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnoptimizer_delete(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, train_with_generator_01_p) {
  int status = ML_ERROR_NONE;

  ml_nnmodel_h model;
  ml_nnlayer_h layers[2];
  ml_nnopt_h optimizer;

  status = ml_nnmodel_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[0], ML_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status =
    ml_nnlayer_set_property(layers[0], "input_shape= 16:1:1:62720",
                            "normalization=true", "bias_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_create(&layers[1], ML_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnlayer_set_property(layers[1], "unit= 10", "activation=softmax",
                                   "bias_zero=true", "weight_decay=l2norm",
                                   "weight_decay_lambda=0.005",
                                   "weight_ini=xavier_uniform", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_create(&optimizer, "adam");
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnoptimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_compile(model, optimizer, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnmodel_train_with_generator(
    model, getMiniBatch_train, getMiniBatch_val, NULL, "epochs=1",
    "batch_size=16", "buffer_size=100", "model_file=model.bin", NULL);

  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnlayer_delete(layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_nnoptimizer_delete(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_nnmodel_destruct(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
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
