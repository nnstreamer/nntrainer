// SPDX-License-Identifier: Apache-2.0
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

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */
/**
 * @brief Tizen ML feature.
 */
#define ML_TRAIN_FEATURE_PATH "tizen.org/feature/machine_learning.training"

/// please note that this is a improvised measure for nnstreamer/api#110
/// proper way will need exposing this particular function in ml-api side
#if (TIZENVERSION >= 7) && (TIZENVERSION < 9999)

/**
 * @brief tizen set feature state from ml api side
 *
 * @param state -1 NOT checked yet, 0 supported, 1 not supported
 * @return int 0 if success
 */
int _ml_tizen_set_feature_state(ml_feature_e feature, int state);
#define ml_api_set_feature_state(...) _ml_tizen_set_feature_state(__VA_ARGS__)

#elif (TIZENVERSION >= 6) && (TIZENVERSION < 9999)

/**
 * @brief tizen set feature state from ml api side
 *
 * @param state -1 NOT checked yet, 0 supported, 1 not supported
 * @return int 0 if success
 */
int ml_tizen_set_feature_state(ml_feature_e feature, int state);
#define ml_api_set_feature_state(...) ml_tizen_set_feature_state(__VA_ARGS__)

#elif (TIZENVERSION <= 5)

#warning Tizen version under 5 does not support setting features for unittest
#define ml_api_set_feature_state(...)

#else /* TIZENVERSION */
#error Tizen version is not defined.
#endif /* TIZENVERSION */

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
void ml_train_tizen_set_feature_state(ml_feature_e feature,
                                      feature_state_t state) {
  pthread_mutex_lock(&feature_info.mutex);

  ml_api_set_feature_state(feature, (int)state);

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
        ml_train_tizen_set_feature_state(ML_FEATURE_TRAINING, NOT_SUPPORTED);
        return ML_ERROR_NOT_SUPPORTED;
      }

      ml_train_tizen_set_feature_state(ML_FEATURE_TRAINING, SUPPORTED);
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

#ifdef __cplusplus
}
#endif /* __cplusplus */
