// SPDX-License-Identifier: Apache-2.0-only
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
      dim[i] = 0;
    }
    len = 0;
    feature_len = 0;
  }

  TensorDim(unsigned int b, unsigned int c, unsigned int h, unsigned int w) :
    TensorDim() {
    setTensorDim(0, b);
    setTensorDim(1, c);
    setTensorDim(2, h);
    setTensorDim(3, w);
    feature_len = c * h * w;
    len = b * feature_len;
  }

  TensorDim(const TensorDim &rhs) = default;

  ~TensorDim(){};

  /**
   *  @brief  Move constructor of Conv 2D Layer.
   *  @param[in] Conv2dLayer &&
   */
  TensorDim(TensorDim &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Optimizer to be moved.
   */
  TensorDim &operator=(TensorDim &&rhs) noexcept;

  /**
   * @brief  swap variable of Conv2D Layer
   * @parma[out] lhs Optimizer
   * @parma[in] rhs Optimizer
   */
  void swap(TensorDim &lhs, TensorDim &rhs) noexcept;

  unsigned int batch() const { return dim[0]; };
  unsigned int channel() const { return dim[1]; };
  unsigned int height() const { return dim[2]; };
  unsigned int width() const { return dim[3]; };
  unsigned int getDataLen() const { return len; };
  unsigned int getFeatureLen() const { return feature_len; };

  void resetLen();
  void batch(unsigned int b) { setTensorDim(0, b); }
  void channel(unsigned int c) { setTensorDim(1, c); }
  void height(unsigned int h) { setTensorDim(2, h); }
  void width(unsigned int w) { setTensorDim(3, w); }

  const unsigned int *getDim() const { return dim; }
  unsigned int getNumDim() const { return MAXDIM; }

  unsigned int getTensorDim(unsigned int idx);
  void setTensorDim(unsigned int idx, unsigned int value);
  int setTensorDim(std::string input_shape);

  TensorDim &operator=(const TensorDim &rhs);
  bool operator==(const TensorDim &rhs) const;
  bool operator!=(const TensorDim &rhs) const { return !(*this == rhs); }
  bool isEmpty() const { return len == 0; }
  unsigned int rank() const;

private:
  unsigned int dim[MAXDIM];
  unsigned int len;
  unsigned int feature_len;
};

std::ostream &operator<<(std::ostream &out, TensorDim const &d);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_DIM_H__ */
