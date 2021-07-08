// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   databuffer_util.h
 * @date   12 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Databuffer utility file.
 *
 */

#define SET_VALIDATION(val)                                               \
  do {                                                                    \
    validation[static_cast<int>(DatasetDataUsageType::DATA_TRAIN)] = val; \
    validation[static_cast<int>(DatasetDataUsageType::DATA_VAL)] = val;   \
    validation[static_cast<int>(DatasetDataUsageType::DATA_TEST)] = val;  \
  } while (0)

#define NN_EXCEPTION_NOTI(val)                             \
  do {                                                     \
    switch (type) {                                        \
    case DatasetDataUsageType::DATA_TRAIN: {               \
      std::lock_guard<std::mutex> lgtrain(readyTrainData); \
      trainReadyFlag = val;                                \
      cv_train.notify_all();                               \
    } break;                                               \
    case DatasetDataUsageType::DATA_VAL: {                 \
      std::lock_guard<std::mutex> lgval(readyValData);     \
      valReadyFlag = val;                                  \
      cv_val.notify_all();                                 \
    } break;                                               \
    case DatasetDataUsageType::DATA_TEST: {                \
      std::lock_guard<std::mutex> lgtest(readyTestData);   \
      testReadyFlag = val;                                 \
      cv_test.notify_all();                                \
    } break;                                               \
    default:                                               \
      break;                                               \
    }                                                      \
  } while (0)
