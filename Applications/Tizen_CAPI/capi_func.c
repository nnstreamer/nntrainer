/* SPDX-License-Identifier: Apache-2.0-only */
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file	capi_func.c
 * @date	01 June 2020
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Classification Example with one FC Layer
 *              The base model for feature extractor is mobilenet v2 with
 * 1280*7*7 feature size. It read the Classification.ini in res directory and
 * run according to the configureation.
 *
 */

#include <limits.h>
#include <nntrainer.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#define num_class 10
#define mini_batch 32
#define feature_size 62720

static bool *duplicate;
static bool *valduplicate;
static bool alloc_train = false;
static bool alloc_val = false;

unsigned int seed;

int gen_data_train(float **outVec, float **outLabel, bool *last);
int gen_data_val(float **outVec, float **outLabel, bool *last);
bool file_exists(const char *filename);

bool file_exists(const char *filename) {
  struct stat buffer;
  return (stat(filename, &buffer) == 0);
}

#define NN_RETURN_STATUS()         \
  do {                             \
    if (status != ML_ERROR_NONE) { \
      return status;               \
    }                              \
  } while (0)

/**
 * @brief     Generate Random integer value between min to max
 * @param[in] min : minimum value
 * @param[in] max : maximum value
 * @retval    min < random value < max
 */
static int range_random(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand_r(&seed);
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] outVec
 * @param[out] outLabel
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
static bool get_data(const char *file_name, float *outVec, float *outLabel,
                     unsigned int id, int file_length) {
  uint64_t position;
  FILE *F;
  unsigned int i;
  size_t ret;

  if (id < 0)
    return false;

  position =
    (uint64_t)((feature_size + num_class) * (uint64_t)id * sizeof(float));
  if (position > file_length) {
    return false;
  }

  F = fopen(file_name, "rb");
  if (F == NULL) {
    printf("Cannot open %s\n", file_name);
    return false;
  }

  if (fseek(F, position, SEEK_SET)) {
    printf("file seek error");
    fclose(F);
    return false;
  }

  for (i = 0; i < feature_size; i++) {
    float f;
    ret = fread((void *)(&f), sizeof(float), 1, F);
    if (!ret) {
      fclose(F);
      return false;
    }
    outVec[i] = f;
  }
  for (i = 0; i < num_class; i++) {
    float f;
    ret = fread((void *)(&f), sizeof(float), 1, F);
    if (!ret) {
      fclose(F);
      return false;
    }
    outLabel[i] = f;
  }

  fclose(F);

  return true;
}

/**
 * @brief      get data which size is mini batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] status for error handling
 * @retval true/false
 */
int gen_data_train(float **outVec, float **outLabel, bool *last) {
  int memI[mini_batch];
  long file_size;
  unsigned int count = 0;
  unsigned int data_size = 0;
  unsigned int i, j;
  FILE *file;
  float *o, *l;

  const char *file_name = "trainingSet.dat";

  if (!file_exists(file_name)) {
    printf("%s does not exists\n", file_name);
    return ML_ERROR_INVALID_PARAMETER;
  }

  file = fopen(file_name, "r");
  fseek(file, 0, SEEK_END);
  file_size = ftell(file);
  fclose(file);
  data_size =
    (unsigned int)(file_size / ((num_class + feature_size) * sizeof(float)));

  if (!alloc_train) {
    duplicate = (bool *)malloc(sizeof(bool) * data_size);
    for (i = 0; i < data_size; ++i) {
      duplicate[i] = false;
    }
    alloc_train = true;
  }

  for (i = 0; i < data_size; i++) {
    if (!duplicate[i])
      count++;
  }

  if (count < mini_batch) {
    if (duplicate == NULL) {
      printf("Error: memory allocation.\n");
      return false;
    }
    free(duplicate);
    alloc_train = false;
    *last = true;
    return ML_ERROR_NONE;
  }

  count = 0;
  while (count < mini_batch) {
    int nomI = range_random(0, data_size - 1);
    if (!duplicate[nomI]) {
      memI[count] = nomI;
      duplicate[nomI] = true;
      count++;
    }
  }

  o = malloc(sizeof(float) * feature_size);
  l = malloc(sizeof(float) * num_class);

  for (i = 0; i < count; i++) {
    get_data(file_name, o, l, memI[i], file_size);

    for (j = 0; j < feature_size; ++j)
      outVec[0][i * feature_size + j] = o[j];
    for (j = 0; j < num_class; ++j)
      outLabel[0][i * num_class + j] = l[j];
  }

  free(o);
  free(l);
  *last = false;
  return ML_ERROR_NONE;
}

/**
 * @brief      get data which size is mini batch for validation
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @retval status for handling error
 */
int gen_data_val(float **outVec, float **outLabel, bool *last) {

  int memI[mini_batch];
  unsigned int i, j;
  unsigned int count = 0;
  unsigned int data_size = 0;
  long file_size;
  float *o, *l;

  const char *file_name = "trainingSet.dat";

  FILE *file = fopen(file_name, "r");
  fseek(file, 0, SEEK_END);
  file_size = ftell(file);
  fclose(file);
  data_size =
    (unsigned int)(file_size / ((num_class + feature_size) * sizeof(float)));

  if (!alloc_val) {
    valduplicate = (bool *)malloc(sizeof(bool) * data_size);
    for (i = 0; i < data_size; ++i) {
      valduplicate[i] = false;
    }
    alloc_val = true;
  }

  for (i = 0; i < data_size; i++) {
    if (!valduplicate[i])
      count++;
  }

  if (count < mini_batch) {
    free(valduplicate);
    alloc_val = false;
    *last = true;
    return ML_ERROR_NONE;
  }

  count = 0;
  while (count < mini_batch) {
    int nomI = range_random(0, data_size - 1);
    if (!valduplicate[nomI]) {
      memI[count] = nomI;
      valduplicate[nomI] = true;
      count++;
    }
  }

  o = malloc(feature_size * sizeof(float));
  l = malloc(num_class * sizeof(float));

  for (i = 0; i < count; i++) {
    get_data(file_name, o, l, memI[i], file_size);

    for (j = 0; j < feature_size; ++j)
      outVec[0][i * feature_size + j] = o[j];
    for (j = 0; j < num_class; ++j)
      outLabel[0][i * num_class + j] = l[j];
  }

  *last = false;

  free(o);
  free(l);

  return ML_ERROR_NONE;
}

int main(int argc, char *argv[]) {

  int status = ML_ERROR_NONE;

  /* handlers for model, layers, optimizer and dataset */
  ml_train_model_h model;
  ml_train_layer_h layers[2];
  ml_train_optimizer_h optimizer;
  ml_train_dataset_h dataset;

  seed = time(NULL);

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
  status = ml_train_model_compile(model, "loss=cross", "batch_size=32", NULL);
  NN_RETURN_STATUS();

  /* create dataset */
  status = ml_train_dataset_create_with_generator(&dataset, gen_data_train,
                                                  gen_data_val, NULL);
  NN_RETURN_STATUS();

  /* set property for dataset */
  status = ml_train_dataset_set_property(dataset, "buffer_size=32", NULL);
  NN_RETURN_STATUS();

  /* set dataset */
  status = ml_train_model_set_dataset(model, dataset);
  NN_RETURN_STATUS();

  /* train model with data files : epochs = 10 and store model file named
   * "model.bin" */
  status = ml_train_model_run(model, "epochs=10", "model_file=model.bin", NULL);
  NN_RETURN_STATUS();

  /* delete model */
  status = ml_train_model_destroy(model);
  NN_RETURN_STATUS();

  return 0;
}
