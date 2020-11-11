// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file	 app_context.cpp
 * @date	 10 November 2020
 * @brief	 This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see		 https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug	   No known bugs except for NYI items
 *
 */
#include <dirent.h>
#include <iostream>
#include <sstream>

#include <app_context.h>
#include <nntrainer_log.h>
#include <util_func.h>

namespace nntrainer {

std::mutex factory_mutex;

AppContext AppContext::instance;

/**
 * @brief initiate global context
 *
 */
static void init_global_context_nntrainer(void) __attribute__((constructor));

/**
 * @brief finialize global context
 *
 */
static void fini_global_context_nntrainer(void) __attribute__((destructor));

static void init_global_context_nntrainer(void) {}

static void fini_global_context_nntrainer(void) {}

AppContext &AppContext::Global() { return AppContext::instance; }

static const std::string func_tag = "[AppContext::setWorkingDirectory] ";

void AppContext::setWorkingDirectory(const std::string &base) {
  DIR *dir = opendir(base.c_str());

  if (!dir) {
    std::stringstream ss;
    ss << func_tag << "path is not directory or has no permission: " << base;
    throw std::invalid_argument(ss.str().c_str());
  }
  closedir(dir);

  char *ret = realpath(base.c_str(), nullptr);

  if (ret == nullptr) {
    std::stringstream ss;
    ss << func_tag << "failed to get canonical path for the path: ";
    throw std::invalid_argument(ss.str().c_str());
  }

  working_path_base = std::string(ret);
  ml_logd("working path base has set: %s", working_path_base.c_str());
  free(ret);
}

const std::string AppContext::getWorkingPath(const std::string &path) {

  /// if path is absolute, return path
  if (path[0] == '/') {
    return path;
  }

  if (working_path_base == std::string()) {
    return path == std::string() ? "." : path;
  }

  return path == std::string() ? working_path_base
                               : working_path_base + "/" + path;
}

} // namespace nntrainer
