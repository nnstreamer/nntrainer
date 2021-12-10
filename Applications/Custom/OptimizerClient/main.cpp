// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   10 December 2021
 * @brief  This file contains the execution part of Optimizer Client example
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <iostream>

#include <app_context.h>
#include <momentum.h>
#include <optimizer.h>

int main() {
  try {
    auto &app_context = nntrainer::AppContext::Global();
    /// registering custom optimizer for nntrainer to understand here
    /// @see app_context::registerFactory for the detail
    app_context.registerFactory(ml::train::createOptimizer<custom::Momentum>,
                                custom::Momentum::type);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  return 0;
}
