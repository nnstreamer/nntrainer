/**
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 */
/**
 * @file nntrainer_error.h
 * @date 03 April 2020
 * @brief NNTrainer Error Codes
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNTRAINER_ERROR_H__
#define __NNTRAINER_ERROR_H__

#if defined(__TIZEN__)
#include <tizen_error.h>
#define ML_ERROR_NONE TIZEN_ERROR_NONE
#define ML_ERROR_INVALID_PARAMETER TIZEN_ERROR_INVALID_PARAMETER
#define ML_ERROR_UNKNOWN TIZEN_ERROR_UNKNOWN
#define ML_ERROR_TIMED_OUT TIZEN_ERROR_TIMED_OUT
#define ML_ERROR_NOT_SUPPORTED TIZEN_ERROR_NOT_SUPPORTED
#define ML_ERROR_PERMISSION_DENIED TIZEN_ERROR_PERMISSION_DENIED
#define ML_ERROR_OUT_OF_MEMORY TIZEN_ERROR_OUT_OF_MEMORY
#define ML_ERROR_CANNOT_ASSIGN_ADDRESS TIZEN_ERROR_CANNOT_ASSIGN_ADDRESS
#define ML_ERROR_BAD_ADDRESS TIZEN_ERROR_BAD_ADDRESS
#else
#include <errno.h>
#define ML_ERROR_NONE (0)
#define ML_ERROR_INVALID_PARAMETER (-EINVAL)
#define ML_ERROR_UNKNOWN (-1073741824LL)
#define ML_ERROR_TIMED_OUT (TIZEN_ERROR_UNKNOWN + 1)
#define ML_ERROR_NOT_SUPPORTED (TIZEN_ERROR_UNKNOWN + 2)
#define ML_ERROR_PERMISSION_DENIED (-EACCES)
#define ML_ERROR_OUT_OF_MEMORY (-ENOMEM)
#define ML_ERROR_CANNOT_ASSIGN_ADDRESS (-EADDRNOTAVAIL)
#define ML_ERROR_BAD_ADDRESS (-EFAULT)
#endif

#endif /* __NNTRAINER_ERROR_H__ */
