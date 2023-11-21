// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	src_shared_tensor.h
 * @date	16 November 2023
 * @brief	This is a SrcSharedTensor class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __SRC_SHARED_TENSOR_H__
#define __SRC_SHARED_TENSOR_H__
#ifdef __cplusplus

namespace nntrainer {

/**
 * @brief     Enumeration of Weight Initialization Type
 * @todo      support intialization from file
 * @note      Remove this enum class
 */
enum class Initializer {
  ZEROS,          /** Zero initialization */
  ONES,           /** One initialization */
  LECUN_NORMAL,   /** LeCun normal initialization */
  LECUN_UNIFORM,  /** uniform initialization */
  XAVIER_NORMAL,  /** Xavier normal initialization */
  XAVIER_UNIFORM, /** Xavier uniform initialization */
  HE_NORMAL,      /** He normal initialization */
  HE_UNIFORM,     /** He uniform initialization */
  NONE            /** No initialization */
};

/**
 * @class SrcSharedTensorV2
 * @brief Source of the shared tensor
 */
template <typename TensorType> class SrcSharedTensorV2 {
public:
  /**
   * @brief   Constructor for the class
   */
  SrcSharedTensorV2() : src(nullptr), off(0) {}

  /**
   * @brief   Constructor for the class
   * @param[in] tensor source tensor
   * @param[in] offset offset
   */
  SrcSharedTensorV2(const TensorType *tensor, size_t offset) :
    src(tensor),
    off(offset) {}

  /**
   * @brief   Get the allocated src tensor
   */
  const TensorType *tensor() const {
    if (!src)
      throw std::runtime_error("Accessing empty src tensor");

    return src;
  }

  /**
   * @brief   Get the offset from the source tensor
   */
  size_t offset() const { return off; }

private:
  const TensorType *src; /**< Tensor of the source */
  size_t off;            /**< offset from the source data ptr */
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __SRC_SHARED_TENSOR_H__ */
