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

/**
 * @brief Tensor Dimension. This class is used to save dimension information
 *
 */
class TensorDim {
public:
  /**
   * @brief Construct a new Tensor Dim object
   *
   */
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

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param b batch
   * @param c channel
   * @param h height
   * @param w width
   */
  TensorDim(unsigned int b, unsigned int c, unsigned int h, unsigned int w) :
    TensorDim() {
    setTensorDim(0, b);
    setTensorDim(1, c);
    setTensorDim(2, h);
    setTensorDim(3, w);
    feature_len = c * h * w;
    len = b * feature_len;
  }

  /**
   * @brief Copy construct a new tensor dim
   *
   * @param rhs tensor dim to copy from
   */
  TensorDim(const TensorDim &rhs) = default;

  /**
   * @brief Construct a new Tensor Dim object
   *
   * @param shape shape of format N:C:H:W
   */
  TensorDim(const std::string &shape);

  /**
   * @brief Destroy the Tensor Dim object
   *
   */
  ~TensorDim() = default;

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

  /**
   * @brief get batch (axis 0)
   *
   * @return unsigned int batch size
   */
  unsigned int batch() const { return dim[0]; };

  /**
   * @brief get channel (axis 1)
   *
   * @return unsigned int channel size
   */
  unsigned int channel() const { return dim[1]; };

  /**
   * @brief get height (axis 2)
   *
   * @return unsigned int height size
   */
  unsigned int height() const { return dim[2]; };

  /**
   * @brief get width (axis 3)
   *
   * @return unsigned int width size
   */
  unsigned int width() const { return dim[3]; };

  /**
   * @brief Get the Data Len object
   *
   * @return unsigned int get length of the data
   */
  unsigned int getDataLen() const { return len; };

  /**
   * @brief Get the Feature Len object
   *
   * @return unsigned int get feature length
   */
  unsigned int getFeatureLen() const { return feature_len; };

  /**
   * @brief set batch (axis 0)
   *
   * @param b batch to set
   */
  void batch(unsigned int b) { setTensorDim(0, b); }

  /**
   * @brief set channel (axis 1)
   *
   * @param c channel to set
   */
  void channel(unsigned int c) { setTensorDim(1, c); }

  /**
   * @brief set height (axis 2)
   *
   * @param h height to set
   */
  void height(unsigned int h) { setTensorDim(2, h); }

  /**
   * @brief set width (axis 3)
   *
   * @param w width to set
   */
  void width(unsigned int w) { setTensorDim(3, w); }

  /**
   * @brief Get the Dim object
   *
   * @return const unsigned int* array of size[MAXDIM]
   */
  const unsigned int *getDim() const { return dim; }

  /**
   * @brief Get the Num Dim object
   *
   * @return unsigned int fixed value of MAXDIM
   */
  unsigned int getNumDim() const { return MAXDIM; }

  /**
   * @brief calculate tranposed dimension
   * @note In this function, batch direction is not considered, so channel is 0
   * @todo make batch 0
   *
   * @param direction  direction to transpose
   * @return TensorDim calculated dimension
   */
  TensorDim transpose(const std::string &direction) const;

  /**
   * @brief calculate trasposed dimension
   * @note In this function, batch direction is considered 0
   *
   * @param axes axes to be transposed
   * @return TensorDim calculated dimension
   */
  TensorDim transpose(const std::array<unsigned int, MAXDIM> &axes) const;

  /**
   * @brief Get the Tensor dimension for an axis
   *
   * @param idx axis to get
   * @return const unsigned int dimension of the given axis
   */
  const unsigned int getTensorDim(unsigned int idx) const;

  /**
   * @brief Set the Tensor Dim object
   *
   * @param idx axis to set
   * @param value value to set
   */
  void setTensorDim(unsigned int idx, unsigned int value);

  /**
   * @brief Set the Tensor Dim object
   *
   * @param input_shape input_shape of format `N:C:H:W`
   * @return int ML_ERROR_NONE if successs
   */
  int setTensorDim(const std::string &input_shape);

  /**
   * @brief copy assign a dimension
   *
   * @param rhs other side to copy assign
   * @return TensorDim& tensor dimension
   */
  TensorDim &operator=(const TensorDim &rhs);

  /**
   * @brief check if tensor dims are equal
   *
   * @param rhs other side to compare
   * @return true equal
   * @return false not equal
   */
  bool operator==(const TensorDim &rhs) const;

  /**
   * @brief check if tensor dims are not equal
   *
   * @param rhs other side to compare
   * @return true not equal
   * @return false equal
   */
  bool operator!=(const TensorDim &rhs) const { return !(*this == rhs); }

  /**
   * @brief check if given tensor dimension is empty
   *
   * @return true empty
   * @return false not empty
   */
  bool isEmpty() const { return len == 0; }

  /**
   * @brief get index rank (dimension of 1 is considered not valid here)
   *
   * @return unsigned int calculated index
   */
  unsigned int rank() const;

  /**
   * @brief operator[] to get index from tensor_dim
   *
   * @param index index
   * @return unsigned int& returned index reference
   */
  unsigned int &operator[](const unsigned int index);

  /**
   * @brief operator[] to get index from tensor_dim
   *
   * @param index index
   * @return const unsigned int& returned index reference
   */
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
  /**
   * @brief reset length
   *
   */
  void resetLen();

  unsigned int dim[MAXDIM]; /**< underlying dimension type */
  unsigned int len;         /**< number of elements */
  unsigned int feature_len; /**< number of feature lements */
};

/**
 * @brief operator<< to print TensorDim
 *
 * @param out ostream
 * @param d dimension to print
 * @return std::ostream& ostream
 */
std::ostream &operator<<(std::ostream &out, TensorDim const &d);

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_DIM_H__ */
