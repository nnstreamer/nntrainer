// SPDX-License-Identifier: Apache-2.0-only
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file nntrainer_util.cpp
 * @date 10 July 2020
 * @brief NNTrainer/Utilizer C-API Wrapper.
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <nntrainer_internal.h>

/**
 * @brief Convert nntrainer API optimizer type to neural network optimizer type
 */
nntrainer::OptType
ml_optimizer_to_nntrainer_type(ml_train_optimizer_type_e type) {
  switch (type) {
  case ML_TRAIN_OPTIMIZER_TYPE_ADAM:
    return nntrainer::OptType::adam;
  case ML_TRAIN_OPTIMIZER_TYPE_SGD:
    return nntrainer::OptType::sgd;
  default:
    return nntrainer::OptType::unknown;
  }
}
