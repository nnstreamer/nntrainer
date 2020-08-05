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

float random() { return dist(rng); }

float sqrtFloat(float x) { return sqrt(x); };

double sqrtDouble(double x) { return sqrt(x); };

float logFloat(float x) { return log(x + 1.0e-20); }

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
  output.setZero();

  for (unsigned int j = 0; j < c; ++j) {
    for (unsigned int k = 0; k < padding[0]; ++k) {
      for (unsigned int l = 0; l < width_p; ++l) {
        output.setValue(0, j, k, l, 0.0f);
        output.setValue(0, j, k + height_p_h, l, 0.0f);
      }
    }

    for (unsigned int l = 0; l < padding[1]; ++l) {
      for (unsigned int k = padding[0]; k < h; ++k) {
        output.setValue(0, j, k, l, 0.0f);
        output.setValue(0, j, k, l + width_p_h, 0.0f);
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
  Tensor output(in.batch(), in.channel(), in.height() - padding[0] * 2,
                in.width() - padding[1] * 2);
  output.setZero();

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

Tensor rotate_180(Tensor in) {
  Tensor output(in.getDim());
  output.setZero();
  for (unsigned int i = 0; i < in.batch(); ++i) {
    for (unsigned int j = 0; j < in.channel(); ++j) {
      for (unsigned int k = 0; k < in.height(); ++k) {
        for (unsigned int l = 0; l < in.width(); ++l) {
          output.setValue(
            i, j, k, l,
            in.getValue(i, j, (in.height() - k - 1), (in.width() - l - 1)));
        }
      }
    }
  }
  return output;
}

bool isFileExist(std::string file_name) {
  std::ifstream infile(file_name);
  return infile.good();
}

} // namespace nntrainer
