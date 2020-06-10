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
    len = 1;
    feature_len = 1;
  }
  ~TensorDim(){};
  unsigned int batch() const { return dim[0]; };
  unsigned int channel() const { return dim[1]; };
  unsigned int height() const { return dim[2]; };
  unsigned int width() const { return dim[3]; };
  unsigned int getDataLen() const { return len; };
  unsigned int getFeatureLen() const { return feature_len; };

  void resetLen();
  void batch(unsigned int b);
  void channel(unsigned int c);
  void height(unsigned int h);
  void width(unsigned int w);

  unsigned int *getDim() { return dim; }

  int setTensorDim(std::string input_shape);

  void operator=(const TensorDim &from);

private:
  unsigned int dim[4];
  unsigned int len;
  unsigned int feature_len;
};

std::ostream &operator<<(std::ostream &out, TensorDim const &d);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_DIM_H__ */
