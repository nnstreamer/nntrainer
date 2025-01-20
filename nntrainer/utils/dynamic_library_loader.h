// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   dynamic_library_loader.h
 * @date   14 January 2025
 * @brief  Wrapper for loading dynamic libraries on multiple operating systems
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __DYNAMIC_LIBRARY_LOADER__
#define __DYNAMIC_LIBRARY_LOADER__

#include <string>

#ifdef _WIN32
#include "windows.h"

// This flags are not used on windows. Defining those symbols for windows make
// possible using the same external interface for loadLibrary function
#define RTLD_LAZY 0
#define RTLD_NOW 0
#define RTLD_BINDING_MASK 0
#define RTLD_NOLOAD 0
#define RTLD_DEEPBIND 0
#define RTLD_GLOBAL 0
#define RTLD_LOCAL 0
#define RTLD_NODELETE 0

#else
#include <dlfcn.h>
#endif

namespace nntrainer {

/**
 * @brief DynamicLibraryLoader wrap process of loading dynamic libraries for
 * multiple operating system
 *
 */
class DynamicLibraryLoader {
public:
  static void *loadLibrary(const char *path, [[maybe_unused]] const int flag) {
#if defined(_WIN32)
    return LoadLibraryA(path);
#else
    return dlopen(path, flag);
#endif
  }

  static int freeLibrary(void *handle) {
#if defined(_WIN32)
    return FreeLibrary((HMODULE)handle);
#else
    return dlclose(handle);
#endif
  }

  static const char *getLastError() {
#if defined(_WIN32)
    return std::to_string(GetLastError()).c_str();
#else
    return dlerror();
#endif
  }

  static void *loadSymbol(void *handle, const char *symbol_name) {
#if defined(_WIN32)
    return GetProcAddress((HMODULE)handle, symbol_name);
#else
    return dlsym(handle, symbol_name);
#endif
  }
};

} // namespace nntrainer

#endif // __DYNAMIC_LIBRARY_LOADER__
