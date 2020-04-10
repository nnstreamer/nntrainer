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

#include "nntrainer_logger.h"
#include <stdarg.h>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace nntrainer {

/**
 * @brief     logfile name
 */
const char* const Logger::logfile_name = "log_nntrainer_";

/**
 * @brief     instance for single logger
 */
Logger* Logger::ainstance = nullptr;

/**
 * @brief     mutex for lock
 */
std::mutex Logger::smutex;

Logger& Logger::instance() {
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

Logger::Logger() {
  time_t t = time(0);
  struct tm* now = localtime(&t);
  std::stringstream ss;
  ss << logfile_name << std::dec << (now->tm_year + 1900) << std::setfill('0') << std::setw(2) << (now->tm_mon + 1)
     << std::setfill('0') << std::setw(2) << now->tm_mday << std::setfill('0') << std::setw(2) << now->tm_hour
     << std::setfill('0') << std::setw(2) << now->tm_min << std::setfill('0') << std::setw(2) << now->tm_sec << ".out";
  outputstream.open(ss.str(), std::ios_base::app);
  if (!outputstream.good()) {
    throw std::runtime_error("Unable to initialize the Logger!");
  }
}

void Logger::log(const std::string& message, const nntrainer_loglevel loglevel) {
  std::lock_guard<std::mutex> guard(smutex);
  time_t t = time(0);
  struct tm* now = localtime(&t);
  std::stringstream ss;
  switch (loglevel) {
    case NNTRAINER_LOG_INFO:
      ss << "[NNTRAINER INFO  ";
      break;
    case NNTRAINER_LOG_WARN:
      ss << "[NNTRAINER WARNING ";
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

  ss << std::dec << (now->tm_year + 1900) << '-' << std::setfill('0') << std::setw(2) << (now->tm_mon + 1) << '-'
     << std::setfill('0') << std::setw(2) << now->tm_mday << '-' << std::setfill('0') << std::setw(2) << now->tm_hour
     << std::setfill('0') << std::setw(2) << now->tm_min << std::setfill('0') << std::setw(2) << now->tm_sec << ']';

  outputstream << ss.str() << " " << message << std::endl;
}

#ifdef __cplusplus
extern "C" {
#endif
void __nntrainer_log_print(nntrainer_loglevel loglevel, const std::string format_str, ...) {
  int final_n, n = ((int)format_str.size()) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(new char[n]);
    std::strcpy(&formatted[0], format_str.c_str());
    va_start(ap, format_str);
    final_n = vsnprintf(&formatted[0], n, format_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }

  std::string ss = std::string(formatted.get());
  Logger::instance().log(ss, loglevel);

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
