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

#include <assert.h>
#include <math.h>
#include <random>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

static auto rng = [] {
  std::mt19937 rng;
  rng.seed(getSeed());
  return rng;
}();
static std::uniform_real_distribution<float> dist(-0.5, 0.5);

unsigned int getSeed() { return 0; }

Tensor softmaxPrime(Tensor x) {
  int batch = x.getBatch();
  int channel = x.getChannel();
  int width = x.getWidth();
  int height = x.getHeight();
  assert(height == 1);

  Tensor PI = Tensor(x.getDim());

  float *xp = x.getData();
  float *pp = PI.getData();

  for (int k = 0; k < batch; ++k) {
    int K = k * channel * height * width;
    for (int c = 0; c < channel; ++c) {
      int C = K + c * height * width;
      for (int i = 0; i < height; ++i) {
        int I = C + i * width;
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
  }
  return PI;
}

Tensor softmax(Tensor t) {
  int batch = t.getBatch();
  int channel = t.getChannel();
  int height = t.getHeight();
  int width = t.getWidth();
  float *dp;
  float *rp;
  float *tp;

  Tensor result(t.getDim());
  Tensor divisor(t.getDim());

  dp = divisor.getData();
  rp = result.getData();
  tp = t.getData();

  divisor.setZero();

  for (int k = 0; k < batch; k++) {
    int index = k * channel * height * width;
    for (int c = 0; c < channel; c++) {
      index = index + c * height * width;
      float m = std::numeric_limits<float>::lowest();
      // find max
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          if (tp[index + i * width + j] > m)
            m = tp[index + i * width + j];
        }
      }

      // shiftx
      float sum = 0.0;
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          dp[index + width * i + j] = exp(tp[index + i * width + j] - m);
          sum += dp[index + width * i + j];
        }
      }

      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          rp[index + width * i + j] = dp[index + width * i + j] / sum;
        }
      }
    }
  }
  return result;
}

float random() { return dist(rng); }

float sqrtFloat(float x) { return (float)(sqrt(x)); };

float logFloat(float x) { return (float)(log(x)); }

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

float sigmoidePrime(float x) {
  float sprime = sigmoid(x);
  return sprime * (1 - sprime);
}

float tanhFloat(float x) { return (float)tanh(x); }

float tanhPrime(float x) {
  float th = (float)tanh(x);
  return 1.0 - th * th;
}

float relu(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return x;
  }
}

float reluPrime(float x) {
  if (x <= 0.0) {
    return 0.0;
  } else {
    return 1.0;
  }
}

float no_op(float x) { return x; }

// This is 2D zero pad
// TODO : Optimize for multi dimention padding
Tensor zero_pad(int batch, Tensor const &in, unsigned int const *padding) {
  unsigned int c = in.channel();
  unsigned int h = in.height();
  unsigned int w = in.width();

  unsigned int height_p = h + padding[0] * 2;
  unsigned int width_p = w + padding[1] * 2;

  unsigned int height_p_h = h + padding[0];
  unsigned int width_p_h = w + padding[1];

  Tensor output(1, c, height_p, width_p);

  for (unsigned int j = 0; j < c; ++j) {
    for (unsigned int k = 0; k < padding[0]; ++k) {
      for (unsigned int l = 0; l < width_p; ++l) {
        output.setValue(0, j, k, l, 0.0);
        output.setValue(0, j, k + height_p_h, l, 0.0);
      }
    }

    for (unsigned int l = 0; l < padding[1]; ++l) {
      for (unsigned int k = padding[0]; k < h; ++k) {
        output.setValue(0, j, k, l, 0.0);
        output.setValue(0, j, k, l + width_p_h, 0.0);
      }
    }
  }

  for (unsigned int j = 0; j < c; ++j) {
    for (unsigned int k = 0; k < h; ++k) {
      for (unsigned int l = 0; l < w; ++l) {
        output.setValue(0, j, k + padding[0], l + padding[1],
                        in.getValue(batch, j, k, l));
      }
    }
  }

  return output;
}

// This is strip pad and return original tensor
Tensor strip_pad(Tensor const &in, unsigned int const *padding) {
  Tensor output(in.batch(), in.channel(), in.width() - padding[0] * 2,
                in.width() - padding[1] * 2);
  for (unsigned int i = 0; i < in.batch(); ++i) {
    for (unsigned int j = 0; j < in.channel(); ++j) {
      for (unsigned int k = 0; k < output.height(); ++k) {
        for (unsigned int l = 0; l < output.width(); ++l) {
          output.setValue(i, j, k, l,
                          in.getValue(i, j, k + padding[0], l + padding[1]));
        }
      }
    }
  }
  return output;
}

} /* namespace nntrainer */
