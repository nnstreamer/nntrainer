// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   defines.h
 * @date   10 July 2025
 * @brief  Common project defines
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __NNTRAINER_DEFINES_H__
#define __NNTRAINER_DEFINES_H__

#if defined(_WIN32)
#define NNTR_EXPORT __declspec(dllexport)
#else
#define NNTR_EXPORT
#endif

#endif // __NNTRAINER_DEFINES_H__
