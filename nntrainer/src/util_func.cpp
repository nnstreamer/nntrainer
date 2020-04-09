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
 *
 * @file	util_func.cpp
 * @date	08 April 2020
 * @brief	This is collection of math functions
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __UTIL_FUNC_H__
#define __UTIL_FUNC_H__
#include "util_func.h"
#include <assert.h>
#include "math.h"
#include "tensor.h"

Tensors::Tensor softmaxPrime(Tensors::Tensor x) {
  int batch = x.getBatch();
  int width = x.getWidth();
  int height = x.getHeight();
  assert(height == 1);

  Tensors::Tensor PI = Tensors::Tensor(batch, height, width);

  float *xp = x.getData();
  float *pp = PI.getData();

  for (int k = 0; k < batch; ++k) {
    int K = k * height * width;
    for (int i = 0; i < height; ++i) {
      int I = K + i * width;
      for (int j = 0; j < width; ++j) {
        float sum = 0.0;
        for (int l = 0; l < width; ++l) {
          if (j == l) {
            sum += xp[I + l] * (1.0 - xp[I + j]);
          } else {
            sum += xp[I + l] * xp[I + j] * -1.0;
          }
        }
        pp[I + j] = sum;
      }
    }
  }
  return PI;
}

Tensors::Tensor softmax(Tensors::Tensor t) {
  int batch = t.getBatch();
  int height = t.getHeight();
  int width = t.getWidth();
  float *dp;
  float *rp;
  float *tp;

  Tensors::Tensor result(batch, height, width);
  Tensors::Tensor divisor(batch, height, 1);

  dp = divisor.getData();
  rp = result.getData();
  tp = t.getData();

  divisor.setZero();

  for (int k = 0; k < batch; k++) {
    int index = k * height;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        dp[index + i] += exp(tp[k * height * width + i * width + j]);
      }
    }
  }

  for (int k = 0; k < batch; ++k) {
    int index = k * height;
    for (int i = 1; i < height; ++i) {
      dp[index] += dp[index + i];
    }
  }

  for (int k = 0; k < batch; k++) {
    int index = k * height;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        int id = k * height * width + i * width + j;
        rp[id] = exp(tp[id]) / dp[index];
      }
    }
  }

  return result;
}

float random(float x) { return (float)(rand() % 10000 + 1) / 10000 - 0.5; }

float sqrt_float(float x) { return (float)(sqrt(x)); };

float log_float(float x) { return (float)(log(x)); }

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

float sigmoidePrime(float x) { return (float)(1.0 / ((1 + exp(-x)) * (1.0 + 1.0 / (exp(-x) + 0.0000001)))); }

float tanh_float(float x) { return (float)tanh(x); }

float tanhPrime(float x) {
  float th = (float)tanh(x);
  return 1.0 - th * th;
}

float Relu(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return x;
  }
}

float ReluPrime(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return 1.0;
  }
}

#endif
