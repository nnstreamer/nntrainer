// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file nntrainer_error.h
 * @date 03 April 2020
 * @brief NNTrainer Error Codes
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNTRAINER_ERROR_H__
#define __NNTRAINER_ERROR_H__

#include <ml-api-common.h>
#if defined(__TIZEN__)
#include <tizen_error.h>
#define ML_ERROR_BAD_ADDRESS TIZEN_ERROR_BAD_ADDRESS
#define ML_ERROR_RESULT_OUT_OF_RANGE TIZEN_ERROR_RESULT_OUT_OF_RANGE
#else
#include <cerrno>
#define ML_ERROR_BAD_ADDRESS (-EFAULT)
#define ML_ERROR_RESULT_OUT_OF_RANGE (-ERANGE)
#endif

#include <functional>
#include <sstream>
#include <stdexcept>

#define NNTR_THROW_IF(pred, err) \
  if ((pred))                    \
    nntrainer::exception::ErrorNotification<err> {}

#define NNTR_THROW_IF_CLEANUP(pred, err, cleanup_func) \
  if ((pred))                                          \
    nntrainer::exception::ErrorNotification<err> { cleanup_func }

namespace nntrainer {

/// @note underscore_case is used for ::exception to keep in accordance with
/// std::exception
namespace exception {

/**
 * @brief Error Notification class, error is thrown when the class is destroyed.
 * DO NOT use this outside as this contains throwing destructor.
 *
 * @tparam Err Error type that except cstring as an argument.
 */
template <typename Err,
          typename std::enable_if_t<std::is_base_of<std::exception, Err>::value,
                                    Err> * = nullptr>
class ErrorNotification {
public:
  /**
   * @brief Construct a new Error Notification object
   *
   */
  explicit ErrorNotification() : cleanup_func() {}

  /**
   * @brief Construct a new Error Notification object
   *
   * @param cleanup_func_ clean up function
   */
  explicit ErrorNotification(std::function<void()> cleanup_func_) :
    cleanup_func(cleanup_func_) {}

  /**
   * @brief Destroy the Error Notification object, Error is thrown when
   * destroying this
   *
   */
  ~ErrorNotification() noexcept(false) {
    if (cleanup_func) {
      cleanup_func();
    }
    throw Err(ss.str().c_str());
  }

  /**
   * @brief Error notification stream wrapper
   *
   * @tparam T anything that will be delegated to the move
   * @param out out stream
   * @param e anything to pass to the stream
   * @return ErrorNotification<Err>&& self
   */
  template <typename T>
  friend ErrorNotification<Err> &&operator<<(ErrorNotification<Err> &&out,
                                             T &&e) {
    out.ss << std::forward<T>(e);
    return std::move(out);
  }

private:
  std::stringstream ss;
  std::function<void()> cleanup_func;
};

/**
 * @brief derived class of invalid argument to represent specific functionality
 * not supported
 * @note this could be either intended or not yet implemented
 */
struct not_supported : public std::invalid_argument {
  using invalid_argument::invalid_argument;
};

/**
 * @brief derived class of invalid argument to represent permission is denied
 */
struct permission_denied : public std::invalid_argument {
  using invalid_argument::invalid_argument;
};

} // namespace exception

} // namespace nntrainer

#endif /* __NNTRAINER_ERROR_H__ */
