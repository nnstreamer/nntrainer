/* SPDX-License-Identifier: Apache-2.0-only */
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file	tensor_dim.h
 * @date	22 May 2020
 * @brief	This is Tensor Dimension Class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_DIM_H__
#define __TENSOR_DIM_H__
#ifdef __cplusplus

#include <iostream>
#include <memory>
#include <regex>
#include <vector>

namespace nntrainer {

#define MAXDIM 4

class TensorDim {
public:
  TensorDim() {
    for (int i = 0; i < MAXDIM; ++i) {
      dim[i] = 1;
    }
  }
  ~TensorDim(){};
  unsigned int batch() { return dim[0]; };
  unsigned int channel() { return dim[1]; };
  unsigned int height() { return dim[2]; };
  unsigned int width() { return dim[3]; };

  void batch(unsigned int b) { dim[0] = b; };
  void channel(unsigned int c) { dim[1] = c; };
  void height(unsigned int h) { dim[2] = h; };
  void width(unsigned int w) { dim[3] = w; };

  unsigned int *getDim() { return dim; }

  int setTensorDim(std::string input_shape);

  void operator=(const TensorDim & from);

private:
  unsigned int dim[4];
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_DIM_H__ */
