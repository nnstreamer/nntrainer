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
 * @file nntrainer_log.h
 * @date 06 April 2020
 * @brief NNTrainer Logger.
 *        Log Util for NNTrainer
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug No known bugs except for NYI items
 */

#ifndef __NNTRAINER_LOG_H__
#define __NNTRAINER_LOG_H__

#define TAG_NAME "nntrainer"

#if defined(__TIZEN__)
#include <dlog.h>

#define ml_logi(...) dlog_print(DLOG_INFO, TAG_NAME, __VA_ARGS__)

#define ml_logw(...) dlog_print(DLOG_WARN, TAG_NAME, __VA_ARGS__)

#define ml_loge(...) dlog_print(DLOG_ERROR, TAG_NAME, __VA_ARGS__)

#define ml_logd(...) dlog_print(DLOG_DEBUG, TAG_NAME, __VA_ARGS__)

#elif defined(__ANDROID__)
#include <android/log.h>

#define ml_logi(...)                                                           \
  __android_log_print(ANDROID_LOG_INFO, TAG_NAME, __VA_ARGS__)

#define ml_logw(...)                                                           \
  __android_log_print(ANDROID_LOG_WARN, TAG_NAME, __VA_ARGS__)

#define ml_loge(...)                                                           \
  __android_log_print(ANDROID_LOG_ERROR, TAG_NAME, __VA_ARGS__)

#define ml_logd(...)                                                           \
  __android_log_print(ANDROID_LOG_DEBUG, TAG_NAME, __VA_ARGS__)

#else /* Linux distro */
#include <nntrainer_logger.h>

#if !defined(ml_logi)
#define ml_logi(format, ...)                                                   \
  __nntrainer_log_print(NNTRAINER_LOG_INFO, "(%s:%s:%d) " format, __FILE__,    \
                        __func__, __LINE__, ##__VA_ARGS__)
#endif

#if !defined(ml_logw)
#define ml_logw(format, ...)                                                   \
  __nntrainer_log_print(NNTRAINER_LOG_WARN, "(%s:%s:%d) " format, __FILE__,    \
                        __func__, __LINE__, ##__VA_ARGS__)
#endif

#if !defined(ml_loge)
#define ml_loge(format, ...)                                                   \
  __nntrainer_log_print(NNTRAINER_LOG_ERROR, "(%s:%s:%d) " format, __FILE__,   \
                        __func__, __LINE__, ##__VA_ARGS__)
#endif

#if !defined(ml_logd)
#define ml_logd(format, ...)                                                   \
  __nntrainer_log_print(NNTRAINER_LOG_DEBUG, "(%s:%s:%d) " format, __FILE__,   \
                        __func__, __LINE__, ##__VA_ARGS__)
#endif

#endif

#endif /* __NNTRAINER_LOG_H__ */
