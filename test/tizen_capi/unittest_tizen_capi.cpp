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
#include <nntrainer_internal.h>

/**
 * @brief Neural Network Model Contruct / Destruct Test (possitive test )
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_01_p) {
  ml_train_model_h handle;
  int status;
  status = ml_train_model_construct(&handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Destruct Test (negative test )
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_02_n) {
  ml_train_model_h handle = NULL;
  int status;
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Construct (negative test)
 */
TEST(nntrainer_capi_nnmodel, construct_destruct_03_n) {
  int status;
  status = ml_train_model_construct(NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file, config_str);
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, construct_conf_01_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "/test/cannot_find.ini";
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, construct_conf_02_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_03_n.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Input_Shape = 32:1:1:62720", "Input_Shape= 32:1:1:0",
                config_file, config_str);
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Compile Test
 */
TEST(nntrainer_capi_nnmodel, compile_02_n) {
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_03_n.ini";
  status = ml_train_model_compile(NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, compile_05_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layers[2];
  ml_train_layer_h get_layer;
  ml_train_optimizer_h optimizer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape= 32:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on default name */
  status = ml_train_model_get_layer(model, "Input0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(get_layer, layers[0]);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005",
    "weight_ini=xavier_uniform", "name=fc100", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on set name */
  status = ml_train_model_get_layer(model, "fc100", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);
  EXPECT_EQ(get_layer, layers[1]);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, compile_06_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layers[3];
  ml_train_layer_h get_layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape= 32:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer before adding */
  status = ml_train_model_get_layer(model, "Input0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on default name */
  status = ml_train_model_get_layer(model, "Input0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Create another layer with same name, different type */
  status = ml_train_layer_set_property(layers[1], "name=Input0", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Not add layer with existing name */
  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_set_property(layers[1], "name=fc0", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** add layer with different name, different layer type */
  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Find layer based on default name */
  status = ml_train_model_get_layer(model, "fc0", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[2], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** Create another layer with same name, same type */
  status = ml_train_layer_set_property(layers[2], "name=fc0", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** add layer with different name, different layer type */
  status = ml_train_model_add_layer(model, layers[2]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_set_property(layers[2], "name=fc1", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  /** add layer with different name, different layer type */
  status = ml_train_model_add_layer(model, layers[2]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_layer(model, "fc1", &get_layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_train_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Input_Shape = 32:1:1:62720", "Input_Shape=16:1:1:62720",
                config_file, config_str);
  replaceString("minibatch = 32", "minibatch = 16", config_file, config_str);
  replaceString("BufferSize=100", "", config_file, config_str);
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_run(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_02_n) {
  int status = ML_ERROR_NONE;
  status = ml_train_model_run(NULL, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
}

/**
 * @brief Neural Network Model Train Test
 */
TEST(nntrainer_capi_nnmodel, train_03_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_train_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Input_Shape = 32:1:1:62720", "input_shape=32:1:1:62720",
                config_file, config_str);
  replaceString("minibatch = 32", "minibatch = 16", config_file, config_str);
  replaceString("BufferSize=100", "", config_file, config_str);
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_run(handle, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);
  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "input_shape= 32:1:1:6270", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "normalization = true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(layer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_02_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "input_shape= 32:1:1:6270",
                                       "normalization=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_03_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layer, "input_shape= 32:1:1:62720",
                                       "activation=sigmoid", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Add Layer Test
 */
TEST(nntrainer_capi_nnmodel, addLayer_04_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layers[2];

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape= 32:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_destroy(layers[0]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_layer_destroy(layers[1]);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model add layer test
 */
TEST(nntrainer_capi_nnmodel, addLayer_05_n) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model = NULL;
  ml_train_layer_h layer = NULL;

  std::string config_file = "./test_compile_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file, config_str);

  status = ml_train_model_construct_with_conf(config_file.c_str(), &model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layer, ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layer, "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layer);
  EXPECT_EQ(status, ML_ERROR_NOT_SUPPORTED);

  status = ml_train_layer_destroy(layer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, create_optimizer_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_optimizer_h optimizer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_destroy(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, create_optimizer_02_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape= 32:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_destroy(optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, create_optimizer_03_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape= 32:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_destroy(optimizer);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, train_with_file_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;
  ml_train_dataset_h dataset;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape= 16:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005",
    "weight_ini=xavier_uniform", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_file(&dataset, "trainingSet.dat",
                                             "valSet.dat", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "label_data=label.dat",
                                         "buffer_size=100", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, "loss=cross", "batch_size=16", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run(model, "epochs=2", "model_file=model.bin", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Optimizer Test
 */
TEST(nntrainer_capi_nnmodel, train_with_generator_01_p) {
  int status = ML_ERROR_NONE;

  ml_train_model_h model;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;
  ml_train_dataset_h dataset;

  status = ml_train_model_construct(&model);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(layers[0], "input_shape= 16:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[0]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005",
    "weight_ini=xavier_uniform", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_add_layer(model, layers[1]);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.002", "beta2=0.001", "epsilon=1e-7", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_optimizer(model, optimizer);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_create_with_generator(&dataset, getMiniBatch_train,
                                                  getMiniBatch_val, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_dataset_set_property(dataset, "buffer_size=100", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_set_dataset(model, dataset);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_compile(model, "loss=cross", NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_run(model, "epochs=2", "batch_size=16",
                              "model_file=model.bin", NULL);

  status = ml_train_model_destroy(model);
  EXPECT_EQ(status, ML_ERROR_NONE);
}

/**
 * @brief Neural Network Model Summary Test summary verbosity of tensor
 */
TEST(nntrainer_capi_summary, summary_01_p) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file, config_str);
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  char *sum = NULL;
  status = ml_train_model_get_summary(handle, ML_TRAIN_SUMMARY_TENSOR, &sum);
  EXPECT_EQ(status, ML_ERROR_NONE);

  EXPECT_GT(strlen(sum), 100);

  status = ml_train_model_destroy(handle);
  EXPECT_EQ(status, ML_ERROR_NONE);

  delete[] sum;
}

/**
 * @brief Neural Network Model Summary Test summary with tensor
 */
TEST(nntrainer_capi_summary, summary_02_n) {
  ml_train_model_h handle = NULL;
  int status = ML_ERROR_NONE;
  std::string config_file = "./test_compile_01_p.ini";
  RESET_CONFIG(config_file.c_str());
  replaceString("Layers = inputlayer outputlayer",
                "Layers = inputlayer outputlayer", config_file, config_str);
  status = ml_train_model_construct_with_conf(config_file.c_str(), &handle);
  EXPECT_EQ(status, ML_ERROR_NONE);
  status = ml_train_model_compile(handle, NULL);
  EXPECT_EQ(status, ML_ERROR_NONE);

  status = ml_train_model_get_summary(handle, ML_TRAIN_SUMMARY_TENSOR, NULL);
  EXPECT_EQ(status, ML_ERROR_INVALID_PARAMETER);

  status = ml_train_model_destroy(handle);
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
