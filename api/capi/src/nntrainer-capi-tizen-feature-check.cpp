// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file nnstreamer-capi-tizen-feature-check.cpp
 * @date 7 August 2020
 * @brief NNTrainer/C-API Tizen dependent functions.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author MyungJoo Ham <myungjoo.ham@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#if !defined(__TIZEN__) || !defined(__FEATURE_CHECK_SUPPORT__)
#error "This file can be included only in Tizen."
#endif

#include <pthread.h>
#include <system_info.h>

#include <nntrainer_internal.h>

/**
 * @brief Tizen ML feature.
 */
#define ML_TRAIN_FEATURE_PATH "tizen.org/feature/machine_learning.training"

/**
 * @brief Internal struct to control tizen feature support
 * (machine_learning.training). -1: Not checked yet, 0: Not supported, 1:
 * Supported
 */
typedef struct _feature_info_s {
  pthread_mutex_t mutex;
  feature_state_t feature_state;

  _feature_info_s() : feature_state(NOT_CHECKED_YET) {
    pthread_mutex_init(&mutex, NULL);
  }

  ~_feature_info_s() { pthread_mutex_destroy(&mutex); }
} feature_info_s;

static feature_info_s feature_info;

/**
 * @brief Set the feature status of machine_learning.training.
 */
void ml_tizen_set_feature_state(feature_state_t state) {
  pthread_mutex_lock(&feature_info.mutex);

  /**
   * Update feature status
   * -1: Not checked yet, 0: Not supported, 1: Supported
   */
  feature_info.feature_state = state;

  pthread_mutex_unlock(&feature_info.mutex);
}

/**
 * @brief Checks whether machine_learning.training feature is enabled or not.
 */
int ml_tizen_get_feature_enabled(void) {
  int ret;
  int feature_enabled;

  pthread_mutex_lock(&feature_info.mutex);
  feature_enabled = feature_info.feature_state;
  pthread_mutex_unlock(&feature_info.mutex);

  if (NOT_SUPPORTED == feature_enabled) {
    ml_loge("machine_learning.training NOT supported");
    return ML_ERROR_NOT_SUPPORTED;
  } else if (NOT_CHECKED_YET == feature_enabled) {
    bool ml_train_supported = false;
    ret =
      system_info_get_platform_bool(ML_TRAIN_FEATURE_PATH, &ml_train_supported);
    if (0 == ret) {
      if (false == ml_train_supported) {
        ml_loge("machine_learning.training NOT supported");
        ml_tizen_set_feature_state(NOT_SUPPORTED);
        return ML_ERROR_NOT_SUPPORTED;
      }

      ml_tizen_set_feature_state(SUPPORTED);
    } else {
      switch (ret) {
      case SYSTEM_INFO_ERROR_INVALID_PARAMETER:
        ml_loge("failed to get feature value because feature key is not vaild");
        ret = ML_ERROR_NOT_SUPPORTED;
        break;

      case SYSTEM_INFO_ERROR_IO_ERROR:
        ml_loge("failed to get feature value because of input/output error");
        ret = ML_ERROR_NOT_SUPPORTED;
        break;

      case SYSTEM_INFO_ERROR_PERMISSION_DENIED:
        ml_loge("failed to get feature value because of permission denied");
        ret = ML_ERROR_PERMISSION_DENIED;
        break;

      default:
        ml_loge("failed to get feature value because of unknown error");
        ret = ML_ERROR_NOT_SUPPORTED;
        break;
      }
      return ret;
    }
  }

  return ML_ERROR_NONE;
}
