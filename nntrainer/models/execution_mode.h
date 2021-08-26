// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   execution_mode.h
 * @date   25 June 2020
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is mode of executions
 *
 */

#ifndef __EXECUTION_MODE_H__
#define __EXECUTION_MODE_H__

namespace nntrainer {

/**
 * @brief   class telling the execution mode of the model/operation
 */
enum class ExecutionMode {
  TRAIN,     /** Training mode, label is necessary */
  INFERENCE, /** Inference mode, label is optional */
  VALIDATE   /** Validate mode, label is necessary */
};

}; // namespace nntrainer

#endif // __EXECUTION_MODE_H__
