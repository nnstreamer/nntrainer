/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file nntrainer_logger.cpp
 * @date 02 April 2020
 * @brief NNTrainer Logger
 *        This allows to logging nntrainer logs.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cstring>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <nntrainer_logger.h>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <util_func.h>

namespace nntrainer {

/**
 * @brief     logfile name
 */
const char *const Logger::logfile_name = "log_nntrainer_";

/**
 * @brief     instance for single logger
 */
Logger *Logger::ainstance = nullptr;

/**
 * @brief     mutex for lock
 */
std::mutex Logger::smutex;

Logger &Logger::instance() {
  static Cleanup cleanup;

  std::lock_guard<std::mutex> guard(smutex);
  if (ainstance == nullptr)
    ainstance = new Logger();
  return *ainstance;
}

Logger::Cleanup::~Cleanup() {
  std::lock_guard<std::mutex> guard(Logger::smutex);
  delete Logger::ainstance;
  Logger::ainstance = nullptr;
}

Logger::~Logger() { outputstream.close(); }

Logger::Logger() : ts_type(NNTRAINER_LOG_TIMESTAMP_SEC) {
  struct tm now;
  getLocaltime(&now);
  std::stringstream ss;
  ss << logfile_name << std::dec << (now.tm_year + 1900) << std::setfill('0')
     << std::setw(2) << (now.tm_mon + 1) << std::setfill('0') << std::setw(2)
     << now.tm_mday << std::setfill('0') << std::setw(2) << now.tm_hour
     << std::setfill('0') << std::setw(2) << now.tm_min << std::setfill('0')
     << std::setw(2) << now.tm_sec << ".out";
  outputstream.open(ss.str(), std::ios_base::app);
  if (!outputstream.good()) {
    char buf[256];
    char *ret = getcwd(buf, 256);
    std::string cur_path = std::string(buf);
    std::string err_msg =
      "Unable to initialize the Logger on path(" + cur_path + ")";
    throw std::runtime_error(err_msg);
  }
}

void Logger::log(const std::string &message,
                 const nntrainer_loglevel loglevel) {
  std::lock_guard<std::mutex> guard(smutex);
  std::stringstream ss;

  switch (loglevel) {
  case NNTRAINER_LOG_INFO:
    ss << "[NNTRAINER INFO  ";
    break;
  case NNTRAINER_LOG_WARN:
    ss << "[NNTRAINER WARN  ";
    break;
  case NNTRAINER_LOG_ERROR:
    ss << "[NNTRAINER ERROR ";
    break;
  case NNTRAINER_LOG_DEBUG:
    ss << "[NNTRAINER DEBUG ";
    break;
  default:
    break;
  }

  if (ts_type == NNTRAINER_LOG_TIMESTAMP_MS) {
    static auto start = std::chrono::system_clock::now().time_since_epoch();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch() - start)
                .count();

    ss << "[ " << ms << " ]";
  } else if (ts_type == NNTRAINER_LOG_TIMESTAMP_SEC) {
    struct tm now;
    getLocaltime(&now);

    ss << std::dec << (now.tm_year + 1900) << '-' << std::setfill('0')
       << std::setw(2) << (now.tm_mon + 1) << '-' << std::setfill('0')
       << std::setw(2) << now.tm_mday << ' ' << std::setfill('0')
       << std::setw(2) << now.tm_hour << ':' << std::setfill('0')
       << std::setw(2) << now.tm_min << ':' << std::setfill('0') << std::setw(2)
       << now.tm_sec << ']';
  }

  outputstream << ss.str() << " " << message << std::endl;
}

#ifdef __cplusplus
extern "C" {
#endif
void __nntrainer_log_print(nntrainer_loglevel loglevel,
                           const std::string format_str, ...) {
  int final_n, n = ((int)format_str.size()) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);
    std::strncpy(&formatted[0], format_str.c_str(), format_str.size());
    va_start(ap, format_str);
    final_n = vsnprintf(&formatted[0], n, format_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }

  std::string ss = std::string(formatted.get());

#if defined(__LOGGING__)
  Logger::instance().log(ss, loglevel);
#else

  switch (loglevel) {
  case NNTRAINER_LOG_ERROR:
    std::cerr << ss << std::endl;
    break;
  case NNTRAINER_LOG_INFO:
  case NNTRAINER_LOG_WARN:
  case NNTRAINER_LOG_DEBUG:
    std::cout << ss << std::endl;
  default:
    break;
  }

#endif
}

#ifdef __cplusplus
}
#endif
} /* namespace nntrainer */
