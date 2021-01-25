/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer API / Tizen Machine-Learning API Common Header
 * Copyright (C) 2020 MyungJoo Ham <myungjoo.ham@samsung.com>
 */
/**
 * @file	ml-api-common.h
 * @date	07 May 2020
 * @brief	Dummy ML-API Common Header from nnstreamer. Relicensed by authors.
 * @see		https://github.com/nnstreamer/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @details
 *      More entries might be migrated from nnstreamer.h if
 *    other modules of Tizen-ML APIs use common data structures.
 */
#ifndef __ML_API_COMMON_H__
#define __ML_API_COMMON_H__

#ifdef __TIZEN__

#error TIZEN build should not use this file.

#endif /** __TIZEN__ **/

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <errno.h>
#define ML_ERROR_NONE (0)
#define ML_ERROR_INVALID_PARAMETER (-EINVAL)
#define ML_ERROR_UNKNOWN (-0x40000000LL)
#define ML_ERROR_TIMED_OUT (ML_ERROR_UNKNOWN + 1)
#define ML_ERROR_NOT_SUPPORTED (ML_ERROR_UNKNOWN + 2)
#define ML_ERROR_PERMISSION_DENIED (-EACCES)
#define ML_ERROR_OUT_OF_MEMORY (-ENOMEM)

#ifdef __cplusplus
}

#endif /* __cplusplus */
#endif /* __ML_API_COMMON_H__ */
