// SPDX-License-Identifier: The MIT License (MIT)
/**
 * Copyright (c) 2017 Facebook Inc.
 * Copyright (c) 2017 Georgia Institute of Technology
 * Copyright 2019 Google LLC
 */
/**
 * @file   fp16.h
 * @date   03 Nov 2023
 * @brief  This is collection of FP16 and FP32 conversion
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Marat Dukhan <maratek@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FP16_H__
#define __FP16_H__

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#include <cmath>
#include <cstdint>
#elif !defined(__OPENCL_VERSION__)
#include <math.h>
#include <stdint.h>
#endif

#include "defines.h"

namespace nntrainer {

/**
 * @brief convert 32-bit float in bit representation to a value
 *
 * @param w 32-bit float as bits
 * @return float 32-bit float as value
 */
NNTR_EXPORT float fp32_from_bits(uint32_t w);

/**
 * @brief convert 32-bit float value as bits
 *
 * @param f 32-bit float as value
 * @return float 32-bit float as bits
 */
NNTR_EXPORT uint32_t fp32_to_bits(float f);

/**
 * @brief convert a 32-bit float to a 16-bit float in bit representation
 *
 * @param f 32-bit float as value
 * @return uint16_t 16-bit float as bits
 */
NNTR_EXPORT uint16_t compute_fp32_to_fp16(float f);

/**
 * @brief convert a 16-bit float, in bit representation, to a 32-bit float
 *
 * @param h 16-bit float as bits
 * @return float 32-bit float as value
 */
NNTR_EXPORT float compute_fp16_to_fp32(uint16_t h);

} /* namespace nntrainer */

#endif /* __FP16_H__ */
