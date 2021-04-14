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

#include <bitset>
#include <vector>

namespace nntrainer {

constexpr const size_t MAXDIM = 4;

class TensorDim {
public:
  TensorDim(const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000) :
    eff_dim_flag(eff_dim_flag_),
    dyn_dim_flag(dyn_dim_flag_) {
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

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param b batch
   * @param c channel
   * @param h height
   * @param w width
   * @param eff_dim_flag_ dimension bit flag to calculate the dynamic
   * dimension, rightmost is width
   */
  TensorDim(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
            const std::bitset<MAXDIM> &eff_dim_flag_ = 0b1111,
            const std::bitset<MAXDIM> &dyn_dim_flag_ = 0b0000) :
    TensorDim(eff_dim_flag_, dyn_dim_flag_) {
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
   * @brief Set the Dim Flag to retrieve effective dimension
   * @note eg) if dimension 4:1:10:1 should be squeezed to 4:10,
   *       set this to 0b1010, rightmost is width
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void setEffDimFlag(const std::bitset<MAXDIM> &dim_flag_) {
    eff_dim_flag = dim_flag_;
  }

  /**
   * @brief Set the dynamic Dim Flag to retrieve dynamic dimension (that can
   * change during running)
   * @note eg) if dimension 4:1:10:1 should be squeezed to dynamic to batch,
   *       set this to 0b1000, rightmost is width
   *
   * @param dim_flag_ dimension bit to calculate, rightmost is width
   */
  void setDynDimFlag(const std::bitset<MAXDIM> &dim_flag_) {
    dyn_dim_flag = dim_flag_;
  }

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
    std::swap(lhs.eff_dim_flag, rhs.eff_dim_flag);
    std::swap(lhs.dyn_dim_flag, rhs.dyn_dim_flag);
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

  /**
   * @brief Get the Effective Dimension of the current
   * @note dynamic dimension is returned as -1
   *
   * @param dynamic if dimension has to be considering dynamic set this to ture
   * @return std::vector<unsigned int> integer vector
   */
  std::vector<int> getEffectiveDimension(bool dynamic = false) const;

  /**
   * @brief check if tensor is dynamic
   *
   * @return true any of dyn_dim_flag is set
   * @return false none of dyn_dim_flag is set
   */
  bool is_dynamic() const { return dyn_dim_flag.any(); }

private:
  std::bitset<MAXDIM> eff_dim_flag; /**< dimension bit flag to define effective
          dimension size */

  std::bitset<MAXDIM> dyn_dim_flag; /**< dimension bit flag to define
dynamic dimension size */
  unsigned int dim[MAXDIM];
  unsigned int len;
  unsigned int feature_len;
};

std::ostream &operator<<(std::ostream &out, TensorDim const &d);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_DIM_H__ */
