// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file	 app_context.h
 * @date	 10 November 2020
 * @brief	 This file contains app context related functions and classes that
 * manages the global configuration of the current environment
 * @see		 https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __APP_CONTEXT_H__
#define __APP_CONTEXT_H__

#include <string>

namespace nntrainer {

/**
 * @class AppContext contains user-dependent configuration
 * @brief App
 */
class AppContext {
public:
  /**
   * @brief Get Global app context.
   *
   * @return AppContext&
   */
  static AppContext &Global();

  /**
   * @brief Set Working Directory for a relative path. working directory is set
   * canonically
   * @param[in] base base directory
   * @throw std::invalid_argument if path is not valid for current system
   */
  void setWorkingDirectory(const std::string &base);

  /**
   * @brief Get Working Path from a relative or representation of a path
   * strating from @a working_path_base.
   * @param[in] path to make full path
   * @return If absolute path is given, returns @a path
   * If relative path is given and working_path_base is not set, return
   * relative path.
   * If relative path is given and working_path_base has set, return absolute
   * path from current working directory
   */
  const std::string getWorkingPath(const std::string &path = "");

private:
  static AppContext instance;

  std::string working_path_base;
};

} // namespace nntrainer

#endif /* __APP_CONTEXT_H__ */
