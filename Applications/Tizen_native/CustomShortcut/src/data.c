// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file data.c
 * @date 21 Jul 2020
 * @brief TIZEN Native Example App data entry with NNTrainer/CAPI.
 * @see  https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 *
 */
#include "data.h"

#include <string.h>

int parse_route(const char *source, char **route, char **data) {
  char *dst = strdup(source);
  const char sep = ':';
  char *i;
  bool find_data = false;

  if (route == NULL || data == NULL) {
    free(dst);
    return APP_ERROR_INVALID_PARAMETER;
  }

  *route = dst;

  for (i = dst; *i != '\0'; ++i) {
    if (*i == sep) {
      *i = '\0';
      *data = i + 1;
      find_data = true;
      break;
    }
  }

  if (!find_data) {
    *data = NULL;
  }

  return APP_ERROR_NONE;
}

void nntrainer_test() {
  ml_train_model_h model = NULL;

  ml_train_model_construct(model);
  ml_train_model_destroy(model);
}
