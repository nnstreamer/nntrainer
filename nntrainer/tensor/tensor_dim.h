// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file   tensor_dim.h
 * @date   22 May 2020
 * @brief  This is Tensor Dimension Class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_DIM_H__
#define __TENSOR_DIM_H__
#ifdef __cplusplus

#include <array>
#include <iostream>

namespace nntrainer {

constexpr const size_t MAXDIM = 4;

class TensorDim {
public:
  TensorDim() {
    for (size_t i = 0; i < MAXDIM; ++i) {
      dim[i] = 0;
    }
    len = 0;
    feature_len = 0;
  }

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param dims std::initialize_list
   *
   * formats of {w}, {h, w}, {c, h, w}, {b, c, h, w} are accepted
   */
  TensorDim(std::initializer_list<unsigned int> dims) : TensorDim() {
    int shift_size = MAXDIM - dims.size();

    if (shift_size < 0) {
      throw std::invalid_argument("[TensorDim] max dimension is 4");
    }

    unsigned int cnt = 0;

    for (auto &i : dims) {
      setTensorDim(shift_size + cnt, i);
      cnt += 1;
    }
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

  TensorDim(const std::string &shape);

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
  friend void swap(TensorDim &lhs, TensorDim &rhs) noexcept {
    std::swap_ranges(std::begin(lhs.dim), std::begin(lhs.dim) + MAXDIM,
                     std::begin(rhs.dim));
    std::swap(lhs.len, rhs.len);
    std::swap(lhs.feature_len, rhs.feature_len);
  }

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

  const unsigned int getTensorDim(unsigned int idx) const;
  void setTensorDim(unsigned int idx, unsigned int value);
  int setTensorDim(const std::string &input_shape);

  TensorDim &operator=(const TensorDim &rhs);
  bool operator==(const TensorDim &rhs) const;
  bool operator!=(const TensorDim &rhs) const { return !(*this == rhs); }
  bool isEmpty() const { return len == 0; }
  unsigned int rank() const;

  unsigned int &operator[](const unsigned int index);
  const unsigned int &operator[](const unsigned int index) const;

  /**
   * @brief Calculate standard strides
   *
   * @return std::array <int, MAXDIM>
   */
  std::array<unsigned int, MAXDIM> computeStrides() const {
    return {dim[1] * dim[2] * dim[3], dim[2] * dim[3], dim[3], 1};
  }

  /**
   * @brief reverse the dimensions inplace
   */
  void reverse();

private:
  unsigned int dim[MAXDIM];
  unsigned int len;
  unsigned int feature_len;
};

std::ostream &operator<<(std::ostream &out, TensorDim const &d);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_DIM_H__ */
