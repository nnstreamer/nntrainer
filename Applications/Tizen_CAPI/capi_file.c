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
 *
 *
 * @file	capi_file.c
 * @date	19 May 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Classification Example with one FC Layer
 *              The base model for feature extractor is mobilenet v2 with
 * 1280*7*7 feature size. It read the Classification.ini in res directory and
 * run according to the configureation.
 *
 */

#include <nntrainer.h>
#include <stdio.h>

#define NN_RETURN_STATUS()         \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      return status;               \
    }                              \
  } while (0)

int main(int argc, char *argv[]) {

  int status = ML_ERROR_NONE;

  /* handlers for model, layers, optimizer and dataset */
  ml_train_model_h model;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;
  ml_train_dataset_h dataset;

  /* model create */
  status = ml_train_model_construct(&model);
  NN_RETURN_STATUS();
  /* input layer create */
  status = ml_train_layer_create(&layers[0], ML_TRAIN_LAYER_TYPE_INPUT);
  NN_RETURN_STATUS();

  /* set property for input layer */
  status = ml_train_layer_set_property(layers[0], "input_shape= 32:1:1:62720",
                                       "normalization=true",
                                       "bias_init_zero=true", NULL);
  NN_RETURN_STATUS();

  /* add input layer into model */
  status = ml_train_model_add_layer(model, layers[0]);
  NN_RETURN_STATUS();

  /* create fully connected layer */
  status = ml_train_layer_create(&layers[1], ML_TRAIN_LAYER_TYPE_FC);
  NN_RETURN_STATUS();

  /* set property for fc layer */
  status = ml_train_layer_set_property(
    layers[1], "unit= 10", "activation=softmax", "bias_init_zero=true",
    "weight_decay=l2norm", "weight_decay_lambda=0.005",
    "weight_ini=xavier_uniform", NULL);
  NN_RETURN_STATUS();

  /* add fc layer into model */
  status = ml_train_model_add_layer(model, layers[1]);
  NN_RETURN_STATUS();

  /* create optimizer */
  status = ml_train_optimizer_create(&optimizer, ML_TRAIN_OPTIMIZER_TYPE_ADAM);
  NN_RETURN_STATUS();

  /* set property for optimizer */
  status = ml_train_optimizer_set_property(
    optimizer, "learning_rate=0.0001", "decay_rate=0.96", "decay_steps=1000",
    "beta1=0.9", "beta2=0.9999", "epsilon=1e-7", NULL);
  NN_RETURN_STATUS();

  /* set optimizer */
  status = ml_train_model_set_optimizer(model, optimizer);
  NN_RETURN_STATUS();

  /* compile model with cross entropy loss function */
  status = ml_train_model_compile(model, "loss=cross", NULL);
  NN_RETURN_STATUS();

  /* create dataset */
  status = ml_train_dataset_create_with_file(&dataset, "trainingSet.dat",
                                             "valSet.dat", NULL);
  NN_RETURN_STATUS();

  /* set property for dataset */
  status = ml_train_dataset_set_property(dataset, "label_data=label.dat",
                                         "buffer_size=100", NULL);
  NN_RETURN_STATUS();

  /* set dataset */
  status = ml_train_model_set_dataset(model, dataset);
  NN_RETURN_STATUS();

  /* train model with data files : epochs = 10 and store model file named
   * "model.bin" */
  status = ml_train_model_run(model, "epochs=10", "batch_size=32",
                              "model_file=model.bin", NULL);
  NN_RETURN_STATUS();

  /* delete model */
  status = ml_train_model_destroy(model);
  NN_RETURN_STATUS();
  return 0;
}
