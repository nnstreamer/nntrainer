// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   mman_windows.h
 * @date   8 April 2025
 * @brief  Windows implementation of posix mman
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef _MMAN_WINDOWS_H_
#define _MMAN_WINDOWS_H_

#if defined(_WIN32)

#include <cstdint>

#include <sys/types.h>
#include <windows.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PROT_READ 0x1
#define PROT_WRITE 0x2
#define PROT_EXEC 0x4
#define PROT_NONE 0x0

#define MAP_FAILED ((void *)-1)
#define MAP_FILE 0
#define MAP_SHARED 0x01
#define MAP_PRIVATE 0x02
#define MAP_TYPE 0xf
#define MAP_FIXED 0x10
#define MAP_ANONYMOUS 0x20
#define MAP_ANON MAP_ANONYMOUS

#define MS_ASYNC 1
#define MS_SYNC 2
#define MS_INVALIDATE 4

void *mmap(void *addr, size_t len, int prot, int flags, int fd, uint64_t off);
int munmap(void *addr, size_t len);
int msync(void *addr, size_t len, int flags);
int mlock(const void *addr, size_t len);
int munlock(const void *addr, size_t len);

#ifdef __cplusplus
}
#endif

#endif // _WIN32

#endif //  _MMAN_WINDOWS_H_
